import json
import logging
import os

import markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import FileSystemLoader, Environment
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr

from .plotting import _plot_roc_curve, _make_roc_dict, _plot_lift_chart, _make_lift_dict, \
    _make_calibration_dict, _plot_calibration_curve, _plot_score_distribution, \
    _make_importance_dict, _plot_importances, _make_score_thresh_dict, _plot_score_thresh
from ..estimation import BaseEstimator
from ..modeling import BinaryModelFactory
from ..templates import TEMPLATE_DIR, INSIGHTS_REPORT_TEMPLATE_PATH
from ..utils.io import NumpyEncoder, ModelPaths

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class BinaryModelInsights:

    def __init__(self, model_factory: BinaryModelFactory, num_importances=10, importance_sample_size=500):
        self.model_factory = model_factory
        self._analyzed = False
        self.num_importances = num_importances
        self.importance_sample_size = importance_sample_size
        self._model_report = {}
        self.reset()

    def reset(self):
        self._analyzed = False
        self._model_report = {}

    @property
    def model(self):
        return self.model_factory.model

    @property
    def feature_names(self):
        if isinstance(self.model.estimator, Pipeline):
            return self.model_factory.estimator[:-1].get_feature_names_out()
        else:
            return self.model.feature_selector.columns

    @property
    def feature_importances(self):
        self._check_analyzed()
        imp_dict = self._model_report['feature_importance']
        # Turn keys in imp_dict into tuples instead of nested dictionaries. This will give us a multi-index on fields when
        # we have multiple score criteria.
        imp_dict = {(K, k): v for K in imp_dict.keys() for k, v in imp_dict[K].items() if k in ('mean', 'std', 'relative')}
        return pd.DataFrame(imp_dict, index=self.feature_names)

    @property
    def avg_feature_importances(self):
        self._check_analyzed()
        avg = np.mean([v['relative'] for v in self._model_report['feature_importance'].values()], axis=0)
        avg = pd.Series(avg, index=self.feature_names, name='Average Relative Importance')
        avg = avg[avg > 0.0]  # Only get non-zero values
        return avg.sort_values(ascending=False)


    @property
    def n_train_positive(self):
        self._check_analyzed()
        return self._model_report['n_train_positive']

    @property
    def n_train_negative(self):
        self._check_analyzed()
        return self._model_report['n_train_negative']

    @property
    def n_test_positive(self):
        self._check_analyzed()
        return self._model_report['n_test_positive']

    @property
    def n_test_negative(self):
        self._check_analyzed()
        return self._model_report['n_test_negative']

    @property
    def n_features(self):
        self._check_analyzed()
        return self._model_report['n_features']

    @property
    def n_train(self):
        self._check_analyzed()
        return self.n_train_positive + self.n_train_negative

    @property
    def n_test(self):
        self._check_analyzed()
        return self.n_test_positive + self.n_test_negative

    @property
    def test_ratio(self):
        self._check_analyzed()
        return self.n_test / (self.n_train + self.n_test)

    @property
    def gains(self):
        self._check_analyzed()
        return self._model_report['gains']

    @property
    def roc(self):
        self._check_analyzed()
        return self._model_report['roc']

    @property
    def calibration(self):
        self._check_analyzed()
        return self._model_report['calibration']

    @property
    def estimator_pipeline_diagram_html(self):
        if isinstance(self.model.estimator, Pipeline):
            return estimator_html_repr(self.model.estimator)
        else:
            return None

    @property
    def postprocessor_pipeline_diagram_html(self):
        if isinstance(self.model.postprocessor, Pipeline):
            return estimator_html_repr(self.model.postprocessor)
        else:
            return None

    @property
    def random_state(self):
        return self.model_factory.random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Begin evaluating model performance.")
        if not self.model_factory.is_fitted():
            raise ValueError("Model must first be fit before it can be analyzed.")

        logger.debug("Transforming X using fitted feature_selector.")
        X = self.model_factory.feature_selector.transform(X)

        logger.debug("Splitting data into train/test sets.")
        X_train, X_test, y_train, y_test = self.model_factory.train_test_split(X, y)

        logger.debug("Calculating class probabilities on test set using fitted model.")
        scores = self.model.predict_proba(X_test)

        try:
            pos_label = self.model_factory.estimator.classes_[1]
        except AttributeError:
            pos_label = 1

        logger.debug("Calculating count statistics on train and test sets.")
        self._calc_counts(X_train, X_test, y_train, y_test)

        logger.debug("Calculating gains lift chart data.")
        self._model_report['gains'] = _make_lift_dict(y_test, scores)

        logger.debug("Calculating ROC Curve data.")
        self._model_report['roc'] = _make_roc_dict(y_test, scores, pos_label=pos_label)

        logger.debug("Calculating Calibration Curve.")
        self._model_report['calibration'] = _make_calibration_dict(y_test, scores, pos_label=pos_label)

        logger.debug("Calculating score threshold metrics.")
        self._model_report['score_thresh'] = _make_score_thresh_dict(y_test.values, scores[:, 1], pos_label=pos_label)

        # TODO this is awkward. Reword this.
        logger.debug("Calculating full pipeline scores.")
        self._calc_postprocess(X_test, y_test)

        logger.debug("Calculating feature importances")
        self._calc_feature_importance(self.model_factory.estimator, X_test, y_test)

        logger.info("Finished evaluating model performance.")
        self._analyzed = True
        return self

    def analyze(self, X: pd.DataFrame, y: pd.Series):
        return self.fit(X, y)

    def generate_report(self, model_paths: ModelPaths):
        """
        Generate a full-fledged report for the given _model and datasets. This method will create plots for feature
        importances and dependences and embed them to both Markdown and HTML reports that are in the output directory.

        Parameters
        ----------
        output_dir : str
            Output directory for the generated reports.

        Returns
        -------
        report : str
            Markdown formatted string for the report.
        """

        logger.info("Begin generating model insights report.")
        # This line makes plotting available in server environments.
        plt.switch_backend('agg')

        # Create output directories.
        output_dir = os.path.abspath(model_paths.report_folder)
        output_img = os.path.join(output_dir, 'imgs/')
        output_html = os.path.join(output_dir, 'html/')

        os.makedirs(output_img, exist_ok=True)
        os.makedirs(output_html, exist_ok=True)
        logger.debug(f'Report will be output to {output_dir}')
        logger.debug('Generating figures for the report')

        # Generate feature plots and save them to the path.
        logger.debug("Plotting feature importances.")
        n_plots = len(self._model_report['feature_importance'])
        imp_fig = plt.figure(figsize=(12, n_plots*self.num_importances*2 + 5), dpi=200, tight_layout=True)
        self.plot_importances(imp_fig)
        imp_fig.savefig(os.path.join(output_img, 'importances.svg'), format='svg')

        logger.debug("Plotting lift chart.")
        lift_fig = plt.figure(figsize=(9, 9))
        self.plot_lift_chart(lift_fig)
        lift_fig.savefig(os.path.join(output_img, 'lift.svg'), format='svg')

        logger.debug("Plotting ROC curve.")
        roc_fig = plt.figure(figsize=(9, 9))
        self.plot_roc_curve(roc_fig)
        roc_fig.savefig(os.path.join(output_img, 'roc.svg'), format='svg')

        logger.debug("Plotting calibration curve.")
        calibration_fig = plt.figure(figsize=(9, 9))
        self.plot_calibration_curve(calibration_fig)
        calibration_fig.savefig(os.path.join(output_img, 'calibration.svg'), format='svg')

        logger.debug("Plotting output score distribution.")
        scores_fig = plt.figure(figsize=(9, 9))
        self.plot_score_distribution(scores_fig)
        scores_fig.savefig(os.path.join(output_img, 'outdist.svg'), format='svg')

        logger.debug("Plotting score threshold metrics.")
        scores_thresh_fig = plt.figure(figsize=(15, 10), dpi=200, tight_layout=True)
        self.plot_score_thresh(scores_thresh_fig)
        scores_thresh_fig.savefig(os.path.join(output_img, 'score_thresh.svg'), format='svg')

        plt.close('all')

        # Now, we can start putting things into the template.
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        report_template_file = os.path.basename(INSIGHTS_REPORT_TEMPLATE_PATH)
        template = env.get_template(report_template_file)

        report = template.render(insights=self)

        logger.debug("Writing report to file.")
        with open(os.path.join(output_dir, 'report.md'), 'w') as f:
            f.write(report)

        with open(os.path.join(output_dir, 'report.html'), 'w') as f:
            report_html = markdown.markdown(report,
                                            output_format='html5',
                                            extensions=['markdown.extensions.extra'])
            f.write(report_html)

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(self._model_report, f, sort_keys=True, indent=4, cls=NumpyEncoder)

        logger.info(f'Finished generating report. Report is ready at {output_dir}')

        return report

    def plot_roc_curve(self, figure=None):
        _plot_roc_curve(self.roc, figure)

    def plot_lift_chart(self, figure=None):
        _plot_lift_chart(self.gains, figure)

    def plot_calibration_curve(self, figure=None):
        _plot_calibration_curve(self.calibration, figure)

    def plot_score_distribution(self, figure=None):
        _plot_score_distribution(self._model_report['postprocessing'], figure)

    def plot_importances(self, figure=None):
        _plot_importances(self._model_report['feature_importance'], self.feature_names,
                          max_features=self.num_importances, figure=figure)

    def plot_score_thresh(self, figure=None):
        _plot_score_thresh(self._model_report['score_thresh'], figure=figure)

    def _calc_counts(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
        self._model_report['n_train_positive'] = (y_train == 1).sum()
        self._model_report['n_train_negative'] = (y_train != 1).sum()
        self._model_report['n_test_positive'] = (y_test == 1).sum()
        self._model_report['n_test_negative'] = (y_test != 1).sum()
        self._model_report['n_features'] = len(self.feature_names)

    def _check_analyzed(self):
        if not self._analyzed:
            raise ValueError("BinaryModel has not been analyzed. Must run `analyze` before retrieving analysis results")

    def _calc_feature_importance(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
        # reorder feature names to match pipeline order so that calculated importances will be in the correct order
        X_test = X_test[self.feature_names]

        positives = y_test[y_test == 1]
        negatives = y_test[y_test == 0]
        num_pos = len(positives)
        num_neg = len(negatives)

        # Get the number of records to sample for each class. We want the positive and negative class to be balanced.
        num_sample = min(min(num_pos, num_neg), self.importance_sample_size)

        pos_idxs = np.random.choice(positives.index.values, num_sample, replace=False)
        neg_idxs = np.random.choice(negatives.index.values, num_sample, replace=False)
        X_explain = pd.concat((X_test.loc[pos_idxs, :], X_test.loc[neg_idxs, :]), axis=0)
        y_explain = pd.concat((y_test.loc[pos_idxs], y_test.loc[neg_idxs]), axis=0)

        # Wrap a string scoring metric in a list. This enables support for 1 or more scoring metrics.
        if isinstance(self.model_factory.scoring, str):
            scoring_metric = [self.model_factory.scoring]
        else:
            scoring_metric = self.model_factory.scoring
        result = permutation_importance(model, X_explain, y_explain, n_jobs=-1, n_repeats=10, random_state=self.random_state,
                                        scoring=scoring_metric)

        self._model_report['feature_importance'] = _make_importance_dict(result)


    def _calc_postprocess(self, X: pd.DataFrame, y: pd.Series):
        #if self.model.postprocessor is None:
        #    return
        scores = self.model.transform(X)
        self._model_report['postprocessing'] = {'scores': scores}
        #self._model_report['postprocessing'] = {'scores': np.digitize(scores.flatten(), quantiler.quantiles_.flatten(), right=True)}
