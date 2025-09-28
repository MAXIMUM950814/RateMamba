
from mambular.base_models.mambular import Mambular
from mambular.configs.mambular_config import DefaultMambularConfig
from mambular.utils.docstring_generator import generate_docstring
from mambular.models.utils.sklearn_base_classifier import SklearnBaseClassifier
# from .utils.sklearn_base_lss import SklearnBaseLSS
# from .utils.sklearn_base_regressor import SklearnBaseRegressor
class MambularClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultMambularConfig,
        model_description="""
        Mambular classifier. This class extends the SklearnBaseClassifier class and uses the Mambular model
        with the default Mambular configuration.
        """,
        examples="""
        >>> from mambular.models import MambularClassifier
        >>> model = MambularClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)