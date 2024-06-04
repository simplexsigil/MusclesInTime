"""subject - subdataset mapping for AMASS dataset"""

import os.path as osp

dataset_dirs = {
    "BMLmovi": "BMLmovi/BMLmovi",
    "BMLrub": "BMLrub/BioMotionLab_NTroje",
    "EyesJapan": "EyesJapanDataset/Eyes_Japan_Dataset",
    "KIT": "KIT/KIT",
    "TotalCapture": "TotalCapture/TotalCapture",
}

dataset_to_subject = {
    "BMLmovi": [
        "Subject_1_F_MoSh",
        "Subject_2_F_MoSh",
        "Subject_3_F_MoSh",
        "Subject_4_F_MoSh",
        "Subject_5_F_MoSh",
        "Subject_6_F_MoSh",
        "Subject_8_F_MoSh",
        "Subject_9_F_MoSh",
        "Subject_11_F_MoSh",
        "Subject_12_F_MoSh",
        "Subject_13_F_MoSh",
        "Subject_14_F_MoSh",
        "Subject_15_F_MoSh",
        "Subject_16_F_MoSh",
        "Subject_17_F_MoSh",
        "Subject_18_F_MoSh",
        "Subject_19_F_MoSh",
        "Subject_20_F_MoSh",
        "Subject_21_F_MoSh",
        "Subject_22_F_MoSh",
        "Subject_23_F_MoSh",
        "Subject_24_F_MoSh",
        "Subject_25_F_MoSh",
        "Subject_27_F_MoSh",
        "Subject_28_F_MoSh",
        "Subject_29_F_MoSh",
        "Subject_30_F_MoSh",
        "Subject_31_F_MoSh",
        "Subject_32_F_MoSh",
        "Subject_33_F_MoSh",
        "Subject_34_F_MoSh",
        "Subject_35_F_MoSh",
        "Subject_36_F_MoSh",
        "Subject_37_F_MoSh",
        "Subject_38_F_MoSh",
        "Subject_39_F_MoSh",
        "Subject_40_F_MoSh",
        "Subject_41_F_MoSh",
        "Subject_42_F_MoSh",
        "Subject_43_F_MoSh",
        "Subject_44_F_MoSh",
        "Subject_45_F_MoSh",
        "Subject_46_F_MoSh",
        "Subject_47_F_MoSh",
        "Subject_48_F_MoSh",
        "Subject_50_F_MoSh",
        "Subject_52_F_MoSh",
        "Subject_53_F_MoSh",
        "Subject_54_F_MoSh",
        "Subject_55_F_MoSh",
        "Subject_56_F_MoSh",
        "Subject_57_F_MoSh",
        "Subject_58_F_MoSh",
        "Subject_59_F_MoSh",
        "Subject_60_F_MoSh",
        "Subject_61_F_MoSh",
        "Subject_62_F_MoSh",
        "Subject_63_F_MoSh",
        "Subject_64_F_MoSh",
        "Subject_65_F_MoSh",
        "Subject_66_F_MoSh",
        "Subject_67_F_MoSh",
        "Subject_68_F_MoSh",
        "Subject_69_F_MoSh",
        "Subject_70_F_MoSh",
        "Subject_71_F_MoSh",
        "Subject_72_F_MoSh",
        "Subject_73_F_MoSh",
        "Subject_74_F_MoSh",
        "Subject_75_F_MoSh",
        "Subject_76_F_MoSh",
        "Subject_77_F_MoSh",
        "Subject_78_F_MoSh",
        "Subject_79_F_MoSh",
        "Subject_80_F_MoSh",
        "Subject_81_F_MoSh",
        "Subject_82_F_MoSh",
        "Subject_83_F_MoSh",
        "Subject_84_F_MoSh",
        "Subject_85_F_MoSh",
        "Subject_86_F_MoSh",
        "Subject_87_F_MoSh",
        "Subject_88_F_MoSh",
        "Subject_89_F_MoSh",
        "Subject_90_F_MoSh",
    ],
    "BMLrub": [
        "rub001",
        "rub002",
        "rub003",
        "rub004",
        "rub005",
        "rub006",
        "rub007",
        "rub008",
        "rub009",
        "rub010",
        "rub011",
        "rub012",
        "rub014",
        "rub015",
        "rub016",
        "rub017",
        "rub018",
        "rub020",
        "rub021",
        "rub022",
        "rub023",
        "rub024",
        "rub025",
        "rub026",
        "rub027",
        "rub028",
        "rub029",
        "rub030",
        "rub031",
        "rub032",
        "rub033",
        "rub034",
        "rub035",
        "rub036",
        "rub037",
        "rub038",
        "rub039",
        "rub040",
        "rub041",
        "rub042",
        "rub043",
        "rub044",
        "rub045",
        "rub046",
        "rub047",
        "rub048",
        "rub049",
        "rub050",
        "rub051",
        "rub052",
        "rub053",
        "rub054",
        "rub055",
        "rub056",
        "rub057",
        "rub058",
        "rub059",
        "rub060",
        "rub061",
        "rub062",
        "rub063",
        "rub064",
        "rub065",
        "rub066",
        "rub067",
        "rub068",
        "rub069",
        "rub070",
        "rub071",
        "rub072",
        "rub073",
        "rub074",
        "rub075",
        "rub076",
        "rub077",
        "rub078",
        "rub079",
        "rub080",
        "rub081",
        "rub083",
        "rub084",
        "rub085",
        "rub086",
        "rub087",
        "rub088",
        "rub089",
        "rub090",
        "rub091",
        "rub092",
        "rub093",
        "rub094",
        "rub095",
        "rub096",
        "rub097",
        "rub098",
        "rub099",
        "rub100",
        "rub101",
        "rub102",
        "rub103",
        "rub104",
        "rub105",
        "rub106",
        "rub108",
        "rub109",
        "rub110",
        "rub111",
        "rub112",
        "rub113",
        "rub114",
        "rub115",
    ],
    "EyesJapan": [
        "hamada",
        "kaiwa",
        "kanno",
        "kawaguchi",
        "kudo",
        "shiono",
        "takiguchi",
        "yamaoka",
        "yokoyama",
        "ichige",
        "frederic",
        "aita",
    ],
    "KIT": [
        "3",
        "4",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "63",
        "167",
        "183",
        "200",
        "205",
        "291",
        "314",
        "317",
        "348",
        "359",
        "378",
        "379",
        "421",
        "423",
        "424",
        "425",
        "441",
        "442",
        "471",
        "513",
        "551",
        "572",
        "575",
        "576",
        "674",
        "675",
        "725",
        "883",
        "912",
        "917",
        "948",
        "950",
        "952",
        "955",
        "965",
        "969",
        "987",
        "1226",
        "1229",
        "1297",
        "1346",
        "1347",
        "1487",
        "1717",
        "1721",
        "1747",
    ],
    "TotalCapture": [
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
    ],
}

subject_to_dataset = {
    subject: dataset
    for dataset, subjects in dataset_to_subject.items()
    for subject in subjects
}


subject_to_dataset_dir = {
    subject: dataset_dirs[dataset]
    for dataset, subjects in dataset_to_subject.items()
    for subject in subjects
}


def get_feat_p(subject: str, sequence: str) -> str:
    return osp.join(subject_to_dataset_dir[subject], subject, sequence + ".npz")
