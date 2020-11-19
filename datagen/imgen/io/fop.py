import json

def save_json_file(filename, data_dict):
    data_json = json.dumps(data_dict, indent=4)
    save_file(filename, data_json)


def save_file(filename, data):
    with open(filename, "w") as file:
        file.write(data)
        

