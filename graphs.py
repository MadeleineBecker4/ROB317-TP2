import matplotlib.pyplot as plt
import numpy as np
import auxFunctions as af

filename = 'Extrait1-Cosmos_Laundromat1(340p).m4v'
cutGroundTruth = af.getCutGroundTruth(filename) 
nbImages = af.getNbFrame(filename) # nombre de frame de la video
X = np.arange(nbImages)

key_img_mediane = [ 125,  364,  495,  555,  626,  672,  902, 1147, 1245, 1362, 1466, 1541, 1638, 1746, 1822, 1926, 2018, 2106, 2191, 2247, 2360, 2477, 2535, 2598, 2675, 2739, 2801, 2929, 3057, 3112, 3146]
key_img_distmax_naif = [ 249,  478,  510,  599,  652,  690, 1113, 1180, 1309, 1414, 1516, 1546, 1670, 1780, 1863, 1988, 2046, 2165, 2215, 2277, 2340, 2511, 2558, 2574, 2713, 2764, 2837, 2984, 3093, 3130, 3161]
key_img_correl = [ 112,  475,  495,  553,  604,  672,  980, 1172, 1216, 1319, 1485, 1546, 1670, 1733, 1823, 1908, 2026, 2101, 2190, 2232, 2340, 2447, 2530, 2574, 2644, 2755, 2786, 2984, 3050, 3121, 3134]
key_img_chisquare = [ 162,  475,  503,  538,  625,  675,  975, 1167, 1224, 1361, 1465, 1525, 1674, 1733, 1783, 1908, 2044, 2089, 2208, 2239, 2342, 2461, 2530, 2578, 2664, 2756, 2783, 2901, 3041, 3121, 3139]
key_img_intersect = [  31,  475,  490,  533,  604,  673,  974, 1172, 1216, 1361, 1465, 1546, 1670, 1761, 1823, 1908, 2026, 2071, 2190, 2238, 2340, 2447, 2530, 2570, 2664, 2755, 2786, 2931, 3020, 3125, 3134]
key_img_bhattacharyya = [  31,  470,  490,  538,  604,  672,  974, 1172, 1216, 1361, 1485, 1546, 1670, 1761, 1823, 1908, 2044, 2084, 2207, 2238, 2340, 2443, 2530, 2567, 2664, 2755, 2771, 2901, 3020, 3125, 3134]
key_img_hellinger = [  31,  470,  490,  538,  604,  672,  974, 1172, 1216, 1361, 1485, 1546, 1670, 1761, 1823, 1908, 2044, 2084, 2207, 2238, 2340, 2443, 2530, 2567, 2664, 2755, 2771, 2901, 3020, 3125, 3134]
key_img_chisquarealt = [  31,  475,  490,  538,  604,  669,  974, 1172, 1216, 1361, 1485, 1546, 1670, 1761, 1823, 1908, 2044, 2081, 2207, 2238, 2340, 2443, 2530, 2567, 2664, 2755, 2808, 2910, 3020, 3125, 3134]
key_img_kldiv = [  74,  470,  506,  537,  623,  671,  976, 1165, 1221, 1320, 1491, 1547, 1594, 1761, 1823, 1908, 2033, 2122, 2207, 2233, 2340, 2457, 2530, 2567, 2650, 2756, 2771, 2901, 3044, 3121, 3134]


func_mediane = np.zeros(nbImages)
func_distmax_naif = np.zeros(nbImages)
func_distmax = np.zeros(nbImages)
func_chisquare = np.zeros(nbImages)
func_intersect = np.zeros(nbImages)
func_bhattacharyya = np.zeros(nbImages)
func_hellinger = np.zeros(nbImages)
func_chisquarealt = np.zeros(nbImages)
func_kldiv = np.zeros(nbImages)


n = len(key_img_mediane)
for i in range(n):
    func_mediane[key_img_mediane[i]] = 1
    func_distmax_naif[key_img_distmax_naif[i]] = 1
    func_distmax[key_img_correl[i]] = 1
    func_chisquare[key_img_chisquare[i]] = 0.9
    func_intersect[key_img_intersect[i]] = 0.8
    func_bhattacharyya[key_img_bhattacharyya[i]] = 0.7
    func_hellinger[key_img_hellinger[i]] = 0.6
    func_chisquarealt[key_img_chisquarealt[i]] = 0.5
    func_kldiv[key_img_kldiv[i]] = 0.4


for idx in cutGroundTruth:
    plt.axvline(x=idx, color='k')
# plt.plot(X, func_mediane, label = "mediane")
# plt.plot(X, func_distmax_naif, label = "dist max naif")
plt.plot(X, func_distmax, label = "Correlation")
plt.plot(X, func_chisquare, label = "Chi carre")
plt.plot(X, func_intersect, label = "Intersection")
plt.plot(X, func_bhattacharyya, label = "Bhattacharyya")
plt.plot(X, func_hellinger, label = "Hellinger")
plt.plot(X, func_chisquarealt, label = "autre chi carre")
plt.plot(X, func_kldiv, label = "KL Div")
plt.legend()
plt.title("Images clefs par les differentes mesures disponibles")
plt.show()
