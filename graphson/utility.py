from datetime import datetime


def format_timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(int(timestamp/1e9)).strftime('%Y-%m-%d %H:%M:%S')

def string_to_field(value: str):
    try:
        return {
            'type': 'long',
            'value': int(value)
        }
    except ValueError:
        return {
            'type': 'string',
            'value': value
        }

def format_label(model: dict, label_key_list: list[tuple[str, str]]) -> str:
    return ' '.join(
        [ f'{label}: {model.get(key)}' for label, key in label_key_list ]
    )

def sanitize(label: str) -> str:
    return label.replace('\\', '/') \
                .replace('\\"', '\'')