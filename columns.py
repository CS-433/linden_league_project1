def get_columns(split=False):
    """
    Returns the names of numerical and categorical columns.

    Args:
        split (bool): If True, returns a tuple (numerical_columns, categorical_columns).
                      If False, returns a combined list of all columns.

    Returns:
        list or tuple: Depending on 'split' parameter.
    """
    categorical_columns = [
        "HLTHPLN1",
        "MEDCOST",
        "EDUCA",
        "INCOME2",
        "CHOLCHK",
        "PNEUVAC3",
        "EMPLOY1",
        "DIFFWALK",
        "DIFFDRES",
        "DIFFALON",
        "USEEQUIP",
        "_AGEG5YR",
        "GENHLTH",
        "_TOTINDA",
        "_RACE",
        "CVDSTRK3",
        "ASTHMA3",
        "CHCSCNCR",
        "CHCOCNCR",
        "CHCCOPD1",
        "HAVARTH3",
        "ADDEPEV2",
        "CHCKIDNY",
        "_SMOKER3",
        "_RFSMOK3",
        "DRNKANY5",
        "_RFBING5",
        "_RFDRHV5",
        "_RFHYPE5",
        "_RFCHOL",
        "DIABETE3",
    ]

    numerical_columns = [
        "_BMI5",
        "CHILDREN",
        "FTJUDA1_",
        "FRUTDA1_",
        "BEANDAY_",
        "GRENDAY_",
        "ORNGDAY_",
        "VEGEDA1_",
        "PHYSHLTH",
        "MENTHLTH",
        "HTM4",
        "WTKG3",
        "_DRNKWEK",
    ]
    if split:
        return numerical_columns, categorical_columns
    return numerical_columns + categorical_columns


def get_binary_categorical_columns():
    """
    Returns the list of binary categorical columns.

    Returns:
        list: binary categorical column names
    """
    binary_categorical_columns = [
        "HLTHPLN1",
        "MEDCOST",
        "PNEUVAC3",
        "DIFFWALK",
        "DIFFDRES",
        "DIFFALON",
        "USEEQUIP",
        "_TOTINDA",
        "_RACE",
        "CVDSTRK3",
        "ASTHMA3",
        "CHCSCNCR",
        "CHCOCNCR",
        "CHCCOPD1",
        "HAVARTH3",
        "ADDEPEV2",
        "CHCKIDNY",
        "_RFSMOK3",
        "DRNKANY5",
        "_RFBING5",
        "_RFDRHV5",
        "_RFHYPE5",
        "_RFCHOL",
        "DIABETE3",
    ]
    return binary_categorical_columns
