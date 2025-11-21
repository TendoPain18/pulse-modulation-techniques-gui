class DataHandler:
    def __init__(self):
        self.file_names = []
        self.variables = []
        self.data = []

    def add_file_data(self, filename, file_data):
        self.file_names.append(filename)
        variables = [entry[0] for entry in file_data]
        variable_data = [entry[1] for entry in file_data]
        self.variables.extend(variables)
        self.data.extend(variable_data)

    def get_file_variables(self, filename):
        # Return the variables associated with a given filename
        # Modify this based on your actual data structure
        return []