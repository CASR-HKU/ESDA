
def get_sparsity(dataset):
    if "ASL" in dataset:
        return asl_mink_sparsity, asl_mink_kernel
    elif "DVS" in dataset:
        return dvs_mink_sparsity, dvs_mink_kernel
    elif "NMNIST" in dataset:
        return nmnist_mink_sparsity, nmnist_mink_kernel
    elif "NCAL" in dataset:
        return ncal_mink_sparsity, ncal_mink_kernel
    elif "Poker" in dataset:
        return poker_mink_sparsity, poker_mink_kernel
    elif "Roshambo" in dataset:
        return iniRosh_mink_sparsity, iniRosh_mink_kernel
    else:
        raise NotImplementedError


ncal_standard_sparsity = [0.0790, 0.2685, 0.7904, 0.8351, 0.9299, 1.0000]
ncal_mink_sparsity = [0.0790, 0.1789, 0.3457, 0.5491, 0.7075, 0.8334]
ncal_mink_drop_sparsity = [0.0790, 0.1789, 0.2767, 0.4117, 0.5364, 0.6511]

dvs_standard_sparsity = [0.0497, 0.1878, 0.6528, 0.9451, 0.9993, 1.0000]
dvs_mink_sparsity = [0.0497, 0.1131, 0.2521, 0.5113, 0.8297, 0.9857]
dvs_mink_drop_sparsity = [0.0497, 0.1131, 0.2021, 0.3677, 0.5893, 0.7657]

asl_standard_sparsity = [0.0114, 0.0596, 0.2200, 0.4160, 0.7742, 1.0000]
asl_mink_sparsity = [0.0114, 0.0347, 0.0847, 0.1699, 0.2942, 0.4953]
asl_mink_drop_sparsity = [0.0114, 0.0347, 0.0726, 0.1336, 0.2199, 0.3454]

nmnist_standard_sparsity = [0.2284, 0.4471, 0.9088, 1]
nmnist_mink_sparsity = [0.2284, 0.3365, 0.5692, 0.9454]
nmnist_mink_drop_sparsity = [0.2284, 0.3365, 0.4609, 0.7255]

poker_standard_sparsity = [0.0976, 0.2730, 0.7407, 1.0000]
poker_mink_sparsity = [0.0976, 0.2008, 0.3722, 0.6713]
poker_mink_drop_sparsity = [0.0976, 0.2008, 0.3024, 0.5277]

iniRosh_standard_sparsity = [0.0976, 0.2730, 0.7407, 1.0000]
iniRosh_mink_sparsity = [0.1040, 0.2230, 0.2230, 0.3143, 0.4444]
iniRosh_mink_drop_sparsity = [0.0976, 0.2008, 0.3024, 0.5277]


ncal_standard_kernel = [
    [0.2047, 0.1492, 0.1531, 0.1196, 0.1088, 0.1069, 0.0718, 0.051, 0.0349],
    [0.0944, 0.0916, 0.1023, 0.0835, 0.0898, 0.1228, 0.1058, 0.1208, 0.189],
    [0.0022, 0.0048, 0.0558, 0.0038, 0.0044, 0.0771, 0.0112, 0.0073, 0.8334],
    [0.0025, 0.0055, 0.0543, 0.005, 0.006, 0.07, 0.0153, 0.0114, 0.8298],
    [0.0, 0.0003, 0.0176, 0.0004, 0.0012, 0.0555, 0.0023, 0.0076, 0.9152]
]

ncal_mink_kernel = [
    [],
    [0.2465, 0.1561, 0.1391, 0.1167, 0.1034, 0.094, 0.0686, 0.0472, 0.0284],
     [0.1257, 0.1044, 0.1018, 0.0942, 0.0966, 0.1099, 0.1113, 0.1183, 0.1378],
     [0.0334, 0.0418, 0.0583, 0.0577, 0.0705, 0.0949, 0.1079, 0.1542, 0.3813],
     [0.0092, 0.0165, 0.0509, 0.0156, 0.0262, 0.0747, 0.0425, 0.1, 0.6643],
     [0.0017, 0.0034, 0.0719, 0.0033, 0.0074, 0.1171, 0.0071, 0.019, 0.769]
]

ncal_mink_drop_kernel = [
    [0.2465, 0.1561, 0.1391, 0.1167, 0.1034, 0.094, 0.0686, 0.0472, 0.0284],
    [0.1257, 0.1044, 0.1018, 0.0942, 0.0966, 0.1099, 0.1113, 0.1183, 0.1378],
    [0.0417, 0.0511, 0.0664, 0.068, 0.0806, 0.1028, 0.1192, 0.1673, 0.3029],
    [0.0146, 0.024, 0.048, 0.0255, 0.0411, 0.0796, 0.0754, 0.1596, 0.5323],
    [0.0034, 0.01, 0.065, 0.0078, 0.0207, 0.1029, 0.0186, 0.0722, 0.6994]
]


dvs_standard_kernel = [
    [0.4822, 0.1761, 0.0944, 0.0606, 0.0444, 0.0365, 0.0307, 0.0302, 0.0448],
     [0.1937, 0.1839, 0.1284, 0.1086, 0.0853, 0.0739, 0.0631, 0.0612, 0.102],
     [0.01, 0.026, 0.0249, 0.0447, 0.0522, 0.0771, 0.0992, 0.1492, 0.5167],
     [0.0001, 0.0002, 0.0005, 0.0009, 0.002, 0.0053, 0.0119, 0.033, 0.946],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.9999]

]

dvs_mink_kernel = [
    [],
    [0.4822, 0.1761, 0.0944, 0.0606, 0.0444, 0.0365, 0.0307, 0.0302, 0.0448],
    [0.3137, 0.1979, 0.126, 0.0866, 0.0662, 0.0552, 0.0471, 0.0448, 0.0625],
    [0.0811, 0.1075, 0.118, 0.1173, 0.1124, 0.109, 0.1061, 0.108, 0.1405],
    [0.0015, 0.0045, 0.011, 0.0232, 0.0424, 0.0754, 0.1285, 0.2216, 0.492],
    [0.0, 0.0, 0.0, 0.0, 0.0002, 0.0019, 0.0099, 0.0578, 0.9302]
]

dvs_mink_drop_kernel = [
    [0.4822, 0.1761, 0.0944, 0.0606, 0.0444, 0.0365, 0.0307, 0.0302, 0.0448],
     [0.3137, 0.1979, 0.126, 0.0866, 0.0662, 0.0552, 0.0471, 0.0448, 0.0625],
     [0.1081, 0.1298, 0.1307, 0.122, 0.1114, 0.1037, 0.097, 0.0946, 0.1027],
     [0.0055, 0.0146, 0.0296, 0.0532, 0.0826, 0.1229, 0.1689, 0.2287, 0.2939],
     [0.0, 0.0, 0.0, 0.0004, 0.0026, 0.0164, 0.0612, 0.22, 0.6994]
]


asl_standard_kernel = [
    [0.4126, 0.2149, 0.1482, 0.0982, 0.0632, 0.0368, 0.0172, 0.0069, 0.002],
    [0.2134, 0.164, 0.1222, 0.111, 0.1065, 0.1073, 0.0802, 0.0597, 0.0358],
    [0.0079, 0.0166, 0.139, 0.0127, 0.0146, 0.1645, 0.0306, 0.0154, 0.5988],
    [0.0075, 0.0189, 0.0821, 0.0164, 0.0267, 0.1113, 0.0625, 0.0408, 0.6338],
    [0.0001, 0.0018, 0.0023, 0.0055, 0.0032, 0.0521, 0.024, 0.0444, 0.8668]
]

asl_mink_kernel = [
    [],
    [0.4133, 0.2148, 0.1481, 0.098, 0.0631, 0.0367, 0.0171, 0.0069, 0.0019],
    [0.2968, 0.1568, 0.1425, 0.1225, 0.1071, 0.0846, 0.0518, 0.0276, 0.0104],
    [0.26, 0.1204, 0.1053, 0.092, 0.0972, 0.1041, 0.0912, 0.0794, 0.0504],
    [0.1962, 0.1311, 0.1154, 0.0895, 0.0852, 0.0958, 0.0788, 0.0912, 0.1169],
    [0.0551, 0.0791, 0.1049, 0.1136, 0.1223, 0.1527, 0.123, 0.127, 0.1223]
]

asl_mink_drop_kernel = [
    [0.4133, 0.2148, 0.1481, 0.098, 0.0631, 0.0367, 0.0171, 0.0069, 0.0019],
    [0.2968, 0.1568, 0.1425, 0.1225, 0.1071, 0.0846, 0.0518, 0.0276, 0.0104],
    [0.2603, 0.1265, 0.1145, 0.1046, 0.1079, 0.1059, 0.0865, 0.0635, 0.0303],
    [0.2107, 0.1359, 0.1185, 0.0949, 0.0936, 0.0985, 0.0857, 0.0892, 0.0732],
    [0.0802, 0.1059, 0.1288, 0.129, 0.1321, 0.1487, 0.112, 0.0999, 0.0635]

]

nmnist_standard_kernel = [
    [0.2108, 0.1474, 0.116, 0.0773, 0.0669, 0.0816, 0.0595, 0.0749, 0.1655],
    [0.0049, 0.0143, 0.0348, 0.0569, 0.0895, 0.1398, 0.1517, 0.183, 0.325],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0003, 0.0025, 0.0416, 0.9556]
]

nmnist_mink_kernel = [
    [],
    [0.22, 0.1497, 0.1115, 0.0756, 0.0664, 0.0792, 0.0604, 0.0755, 0.1616],
    [0.0089, 0.026, 0.0582, 0.0938, 0.127, 0.1627, 0.159, 0.1635, 0.2009],
    [0.0, 0.0, 0.0, 0.0, 0.0013, 0.0142, 0.0955, 0.3376, 0.5514]
]

nmnist_mink_drop_kernel = [
    [0.2108, 0.1474, 0.116, 0.0773, 0.0669, 0.0816, 0.0595, 0.0749, 0.1655],
    [0.009, 0.0262, 0.0583, 0.0917, 0.1252, 0.1633, 0.1597, 0.1658, 0.2007],
    [0.0, 0.0, 0.0, 0.0003, 0.0057, 0.0429, 0.1848, 0.3949, 0.3715]
]

poker_standard_kernel = [
    [0.1303, 0.108, 0.1178, 0.1056, 0.1187, 0.1262, 0.1098, 0.1014, 0.0823],
    [0.0489, 0.0551, 0.0842, 0.0634, 0.1175, 0.1331, 0.0915, 0.1746, 0.2318],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9]
]

poker_mink_kernel = [
    [],
    [0.1237, 0.1039, 0.1155, 0.106, 0.121, 0.1292, 0.1124, 0.1039, 0.0843],
    [0.046, 0.0496, 0.0763, 0.0714, 0.1356, 0.1283, 0.1295, 0.2034, 0.1598],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.025, 0.525, 0.425]
]

poker_mink_drop_kernel = [
    [0.1237, 0.1039, 0.1155, 0.106, 0.121, 0.1292, 0.1124, 0.1039, 0.0843],
    [0.046, 0.0496, 0.0763, 0.0714, 0.1356, 0.1283, 0.1295, 0.2034, 0.1598],
    [0.0, 0.0, 0.0, 0.0, 0.0125, 0.0375, 0.15, 0.4875, 0.3125]
]

iniRosh_mink_kernel = [
    [0.3331, 0.202, 0.1458, 0.102, 0.075, 0.0557, 0.038, 0.0274, 0.021],
    [0.2001, 0.1313, 0.1318, 0.1215, 0.1081, 0.1018, 0.0821, 0.0665, 0.0568],
    [0.1383, 0.0999, 0.096, 0.0837, 0.0831, 0.1014, 0.0993, 0.1249, 0.1734],
    [0.094, 0.103, 0.1144, 0.1101, 0.1558, 0.1406, 0.1446, 0.1068, 0.0308],
    [0.0023, 0.0135, 0.0248, 0.0631, 0.1014, 0.1509, 0.1622, 0.1892, 0.2928]
]