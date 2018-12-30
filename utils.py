class NestedStruct:
    """
    Class for creating an object from dictionnary
    """
    def __init__(self, **entries):
        for k in entries.keys():
            if isinstance(entries[k], dict):
                entries[k] = NestedStruct(**entries[k])
        self.__dict__.update(entries)
