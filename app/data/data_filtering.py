def filter_data(data):
    return [d for d in data if len(d["output"]) > 50]