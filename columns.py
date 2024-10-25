import numpy as np


def get_all_columns():
    """
    Returns the names of numerical and categorical columns.

    Returns:
        numerical_columns: names of numerical columns
        categorical_columns: names of categorical columns
    """
    categorical_columns = [
        "_STATE",
        "FMONTH",
        "IMONTH",
        "DISPCODE",
        "CTELENUM",
        "PVTRESD1",
        "COLGHOUS",
        "STATERES",
        "CELLFON3",
        "LADULT",
        "CTELNUM1",
        "CELLFON2",
        "CADULT",
        "PVTRESD2",
        "CCLGHOUS",
        "CSTATE",
        "LANDLINE",
        "GENHLTH",
        "HLTHPLN1",
        "PERSDOC2",
        "MEDCOST",
        "CHECKUP1",
        "BPHIGH4",
        "BPMEDS",
        "BLOODCHO",
        "CHOLCHK",
        "TOLDHI2",
        "CVDSTRK3",
        "ASTHMA3",
        "ASTHNOW",
        "CHCSCNCR",
        "CHCOCNCR",
        "CHCCOPD1",
        "HAVARTH3",
        "ADDEPEV2",
        "CHCKIDNY",
        "DIABETE3",
        "SEX",
        "MARITAL",
        "EDUCA",
        "RENTHOM1",
        "NUMHHOL2",
        "NUMPHON2",
        "CPDEMO1",
        "VETERAN3",
        "EMPLOY1",
        "INCOME2",
        "INTERNET",
        "PREGNANT",
        "QLACTLM2",
        "USEEQUIP",
        "BLIND",
        "DECIDE",
        "DIFFWALK",
        "DIFFDRES",
        "DIFFALON",
        "SMOKE100",
        "SMOKDAY2",
        "STOPSMK2",
        "LASTSMK2",
        "USENOW3",
        "EXERANY2",
        "EXRACT11",
        "EXRACT21",
        "LMTJOIN3",
        "ARTHDIS2",
        "ARTHSOCL",
        "SEATBELT",
        "FLUSHOT6",
        "IMFVPLAC",
        "PNEUVAC3",
        "HIVTST6",
        "WHRTST10",
        "PDIABTST",
        "PREDIAB1",
        "INSULIN",
        "EYEEXAM",
        "DIABEYE",
        "DIABEDU",
        "CAREGIV1",
        "CRGVREL1",
        "CRGVLNG1",
        "CRGVHRS1",
        "CRGVPRB1",
        "CRGVPERS",
        "CRGVHOUS",
        "CRGVMST2",
        "CRGVEXPT",
        "VIDFCLT2",
        "VIREDIF3",
        "VIPRFVS2",
        "VINOCRE2",
        "VIEYEXM2",
        "VIINSUR2",
        "VICTRCT4",
        "VIGLUMA2",
        "VIMACDG2",
        "CIMEMLOS",
        "CDHOUSE",
        "CDASSIST",
        "CDHELP",
        "CDSOCIAL",
        "CDDISCUS",
        "WTCHSALT",
        "DRADVISE",
        "ASATTACK",
        "ASYMPTOM",
        "ASNOSLEP",
        "ASTHMED3",
        "ASINHALR",
        "HAREHAB1",
        "STREHAB1",
        "CVDASPRN",
        "ASPUNSAF",
        "RLIVPAIN",
        "RDUCHART",
        "RDUCSTRK",
        "ARTTODAY",
        "ARTHWGT",
        "ARTHEXER",
        "ARTHEDU",
        "TETANUS",
        "HPVADVC2",
        "SHINGLE2",
        "HADMAM",
        "HOWLONG",
        "HADPAP2",
        "LASTPAP2",
        "HPVTEST",
        "HPLSTTST",
        "HADHYST2",
        "PROFEXAM",
        "LENGEXAM",
        "BLDSTOOL",
        "LSTBLDS3",
        "HADSIGM3",
        "HADSGCO1",
        "LASTSIG3",
        "PCPSAAD2",
        "PCPSADI1",
        "PCPSARE1",
        "PSATEST1",
        "PSATIME",
        "PCPSARS1",
        "PCPSADE1",
        "SCNTMNY1",
        "SCNTMEL1",
        "SCNTPAID",
        "SCNTLPAD",
        "SXORIENT",
        "TRNSGNDR",
        "RCSGENDR",
        "RCSRLTN2",
        "CASTHDX2",
        "CASTHNO2",
        "EMTSUPRT",
        "LSATISFY",
        "MISTMNT",
        "ADANXEV",
        "QSTVER",
        "MSCODE",
        "_CHISPNC",
        "_CRACE1",
        "_CPRACE",
        "_DUALUSE",
        "_RFHLTH",
        "_HCVU651",
        "_RFHYPE5",
        "_CHOLCHK",
        "_RFCHOL",
        "_LTASTH1",
        "_CASTHM1",
        "_ASTHMS1",
        "_DRDXAR1",
        "_PRACE1",
        "_MRACE1",
        "_HISPANC",
        "_RACE",
        "_RACEG21",
        "_RACEGR3",
        "_RACE_G1",
        "_AGEG5YR",
        "_AGE65YR",
        "_AGE_G",
        "_BMI5CAT",
        "_RFBMI5",
        "_CHLDCNT",
        "_EDUCAG",
        "_INCOMG",
        "_SMOKER3",
        "_RFSMOK3",
        "DRNKANY5",
        "_RFBING5",
        "_RFDRHV5",
        "_FRTRESP",
        "_VEGRESP",
        "_FRTLT1",
        "_VEGLT1",
        "_FRT16",
        "_VEG23",
        "_FRUITEX",
        "_VEGETEX",
        "_TOTINDA",
        "ACTIN11_",
        "ACTIN21_",
        "PAMISS1_",
        "_PACAT1",
        "_PAINDX1",
        "_PA150R2",
        "_PA300R2",
        "_PA30021",
        "_PASTRNG",
        "_PAREC1",
        "_PASTAE1",
        "_LMTACT1",
        "_LMTWRK1",
        "_LMTSCL1",
        "_RFSEAT2",
        "_RFSEAT3",
        "_FLSHOT6",
        "_PNEUMO2",
        "_AIDTST3",
    ]

    numerical_columns = [
        "SEQNO",
        "FLSHTMY2",
        "_STSTR",
        "_STRWT",
        "_RAWRAKE",
        "_WT2RAKE",
        "_DUALCOR",
        "_PSU",
        "_CLLCPWT",
        "_LLCPWT",
        "IDATE",
        "IDAY",
        "IYEAR",
        "NUMADULT",
        "NUMMEN",
        "NUMWOMEN",
        "HHADULT",
        "PHYSHLTH",
        "MENTHLTH",
        "POORHLTH",
        "DIABAGE2",
        "CHILDREN",
        "WEIGHT2",
        "HEIGHT3",
        "ALCDAY5",
        "AVEDRNK2",
        "DRNK3GE5",
        "MAXDRNKS",
        "FRUITJU1",
        "FRUIT1",
        "FVBEANS",
        "FVGREEN",
        "FVORANG",
        "VEGETAB1",
        "EXEROFT1",
        "EXERHMM1",
        "EXEROFT2",
        "EXERHMM2",
        "STRENGTH",
        "JOINPAIN",
        "HIVTSTD3",
        "BLDSUGAR",
        "FEETCHK2",
        "DOCTDIAB",
        "CHKHEMO3",
        "FEETCHK",
        "LONGWTCH",
        "ASTHMAGE",
        "ASERVIST",
        "ASDRVIST",
        "ASRCHKUP",
        "ASACTLIM",
        "HPVADSHT",
        "PCDMDECN",
        "SCNTWRK1",
        "SCNTLWK1",
        "ADPLEASR",
        "ADDOWN",
        "ADSLEEP",
        "ADENERGY",
        "ADEAT1",
        "ADFAIL",
        "ADTHINK",
        "ADMOVE",
        "QSTLANG",
        "_AGE80",
        "HTIN4",
        "HTM4",
        "WTKG3",
        "_BMI5",
        "DROCDY3_",
        "_DRNKWEK",
        "FTJUDA1_",
        "FRUTDA1_",
        "BEANDAY_",
        "GRENDAY_",
        "ORNGDAY_",
        "VEGEDA1_",
        "_MISFRTN",
        "_MISVEGN",
        "_FRUTSUM",
        "_VEGESUM",
        "METVL11_",
        "METVL21_",
        "MAXVO2_",
        "FC60_",
        "PADUR1_",
        "PADUR2_",
        "PAFREQ1_",
        "PAFREQ2_",
        "_MINAC11",
        "_MINAC21",
        "STRFREQ_",
        "PAMIN11_",
        "PAMIN21_",
        "PA1MIN_",
        "PAVIG11_",
        "PAVIG21_",
        "PA1VIGM_",
    ]

    return numerical_columns, categorical_columns


def get_all_binary_categorical_columns():
    """
    Returns the list of binary categorical columns.

    Returns:
        list: binary categorical column names
    """
    binary_categorical_columns = [
        "DISPCODE",
        "CTELENUM",
        "COLGHOUS",
        "CELLFON3",
        "LADULT",
        "CTELNUM1",
        "CELLFON2",
        "CADULT",
        "PVTRESD2",
        "CCLGHOUS",
        "CSTATE",
        "LANDLINE",
        "PHYSHLTH",
        "MENTHLTH",
        "POORHLTH",
        "HLTHPLN1",
        "MEDCOST",
        "BPMEDS",
        "BLOODCHO",
        "TOLDHI2",
        "CVDSTRK3",
        "ASTHMA3",
        "ASTHNOW",
        "CHCSCNCR",
        "CHCOCNCR",
        "CHCCOPD1",
        "HAVARTH3",
        "ADDEPEV2",
        "CHCKIDNY",
        "DIABAGE2",
        "SEX",
        "NUMHHOL2",
        "CPDEMO1",
        "VETERAN3",
        "CHILDREN",
        "INTERNET",
        "PREGNANT",
        "QLACTLM2",
        "USEEQUIP",
        "BLIND",
        "DECIDE",
        "DIFFWALK",
        "DIFFDRES",
        "DIFFALON",
        "SMOKE100",
        "STOPSMK2",
        "AVEDRNK2",
        "DRNK3GE5",
        "MAXDRNKS",
        "EXERANY2",
        "LMTJOIN3",
        "ARTHDIS2",
        "FLUSHOT6",
        "PNEUVAC3",
        "HIVTST6",
        "PDIABTST",
        "INSULIN",
        "DOCTDIAB",
        "FEETCHK",
        "DIABEYE",
        "DIABEDU",
        "CRGVPERS",
        "CRGVHOUS",
        "CRGVEXPT",
        "CIMEMLOS",
        "CDDISCUS",
        "WTCHSALT",
        "DRADVISE",
        "ASATTACK",
        "HAREHAB1",
        "STREHAB1",
        "CVDASPRN",
        "RLIVPAIN",
        "RDUCHART",
        "RDUCSTRK",
        "ARTHWGT",
        "ARTHEXER",
        "ARTHEDU",
        "HPVADSHT",
        "SHINGLE2",
        "HADMAM",
        "HADPAP2",
        "HPVTEST",
        "HADHYST2",
        "PROFEXAM",
        "BLDSTOOL",
        "HADSIGM3",
        "HADSGCO1",
        "PCPSAAD2",
        "PCPSADI1",
        "PCPSARE1",
        "PSATEST1",
        "RCSGENDR",
        "CASTHDX2",
        "CASTHNO2",
        "ADPLEASR",
        "ADDOWN",
        "ADSLEEP",
        "ADENERGY",
        "ADEAT1",
        "ADFAIL",
        "ADTHINK",
        "ADMOVE",
        "MISTMNT",
        "ADANXEV",
        "_CHISPNC",
        "_RFHLTH",
        "_HCVU651",
        "_RFHYPE5",
        "_RFCHOL",
        "_LTASTH1",
        "_CASTHM1",
        "_DRDXAR1",
        "_HISPANC",
        "_RACEG21",
        "HTIN4",
        "HTM4",
        "WTKG3",
        "_RFBMI5",
        "_RFSMOK3",
        "DRNKANY5",
        "_RFBING5",
        "_RFDRHV5",
        "_MISFRTN",
        "_MISVEGN",
        "_FRTRESP",
        "_VEGRESP",
        "_FRTLT1",
        "_VEGLT1",
        "_FRT16",
        "_VEG23",
        "_TOTINDA",
        "METVL11_",
        "METVL21_",
        "MAXVO2_",
        "FC60_",
        "PAFREQ1_",
        "PAFREQ2_",
        "_MINAC11",
        "_MINAC21",
        "STRFREQ_",
        "PAMISS1_",
        "_PAINDX1",
        "_PA30021",
        "_PASTRNG",
        "_PASTAE1",
        "_RFSEAT2",
        "_RFSEAT3",
        "_FLSHOT6",
        "_PNEUMO2",
        "_AIDTST3",
    ]

    return binary_categorical_columns


def get_selected_columns():
    """
    Returns the names of numerical and categorical columns.

    Returns:
        numerical_columns: names of numerical columns
        categorical_columns: names of categorical columns
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
    return numerical_columns, categorical_columns


def get_selected_binary_categorical_columns():
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


def get_random_column_subset(percentage=100):
    """
    Returns a random subset of all columns (numerical, categorical, binary) based on the specified percentage.

    Args:
        percentage (int): Percentage of columns to return.

    Returns:
        tuple: (selected_numerical_columns, selected_categorical_columns, selected_binary_columns)
    """
    numerical_columns, categorical_columns = get_all_columns()
    binary_categorical_columns = get_all_binary_categorical_columns()

    all_columns = list(set(numerical_columns + categorical_columns))

    num_columns = int(len(all_columns) * (percentage / 100))

    selected_columns = np.random.choice(
        all_columns, num_columns, replace=False
    )

    selected_numerical_columns = [
        col for col in selected_columns if col in numerical_columns
    ]
    selected_categorical_columns = [
        col for col in selected_columns if col in categorical_columns
    ]
    selected_binary_columns = [
        col for col in selected_columns if col in binary_categorical_columns
    ]

    return (
        selected_numerical_columns,
        selected_categorical_columns,
        selected_binary_columns,
    )
