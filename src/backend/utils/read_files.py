def load_labels(filename):
    with open(filename, "r") as f:
        result = [line.strip() for line in f.readlines()]
    print(result)
    return result
