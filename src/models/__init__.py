from .retinanet_ref import ReferenceRetinaNet
from .fasterrcnn_ref import ReferenceFasterRCNN
from .retinanet_custom import CustomRetinaNet
from .fasterrcnn_custom import CustomFasterRCNN

def get_model(model_name: str, config: dict):
    models = {
        "retinanet_ref": ReferenceRetinaNet,
        "faster_rcnn_ref": ReferenceFasterRCNN,
        "retinanet_custom": CustomRetinaNet,
        "faster_rcnn_custom": CustomFasterRCNN
    }

    if model_name not in models:
        raise ValueError(f"Модель '{model_name}' не найдена. Доступные: {list(models.keys())}")

    print(f"[Model Factory] Создание модели: {model_name}")
    return models[model_name](config)