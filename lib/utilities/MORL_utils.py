import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from pymoo.factory import get_performance_indicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def real_pareto_cal(args):
    if args.scenario_name == 'dst-v0':
        time = [-1, -3, -5, -7, -8, -9, -13, -14, -17, -19]
        treasure = [0.7, 8.2, 11.5, 14., 15.1, 16.1, 19.6, 20.3, 22.4, 23.7]
        
        # apply discount
        dst_time = (-(1 - np.power(args.gamma, -np.asarray(time))) / (1 - args.gamma)).tolist()
        dst_treasure = (np.power(args.gamma, -np.asarray(time) - 1) * np.asarray(treasure)).tolist()
        return dst_time, dst_treasure
    if args.scenario_name == 'ftn-v0':
        # Add data
        FRUITS_DICT = {'6': 
                   [[0.26745039, 3.54435815, 4.39088762, 0.5898826, 7.7984232, 2.63110921],
                    [0.46075946, 5.29084735, 7.92804145, 2.28448495, 1.01115855, 1.64300963],
                    [0.5844333, 4.28059796, 7.00237899, 2.51448544, 4.32323182, 2.69974756],
                    [4.01332296, 7.17080888, 1.46983043, 3.82182158, 2.20659648, 3.29195217],
                    [3.74601154, 0.91228863, 5.92072559, 4.37056585, 2.73662976, 4.84656035],
                    [2.42167773, 3.34415377, 6.35216354, 0.03806333, 0.66323198, 6.49313525],
                    [5.26768145, 0.23364916, 0.23646111, 1.25030802, 1.41161868, 8.28161149],
                    [0.19537027, 2.3433365, 6.62653841, 2.84247689, 1.71456358, 6.28809908],
                    [5.9254461, 0.35473447, 5.4459742, 3.57702685, 0.95237377, 4.62628146],
                    [2.22158757, 1.01733311, 7.9499714, 3.6379799, 3.77557594, 1.82692783],
                    [4.43311346, 4.91328158, 5.11707495, 3.9065904, 2.22236853, 3.13406169],
                    [6.44612546, 5.14526023, 1.37156642, 1.37449512, 0.62784821, 5.27343712],
                    [2.39054781, 1.97492965, 4.51911017, 0.07046741, 1.74139824, 8.18077893],
                    [3.26794393, 3.28877157, 2.91598351, 0.49403134, 7.86629258, 2.80694464],
                    [3.96600091, 3.6266905, 4.44655634, 6.0366069, 1.58135473, 3.52204257],
                    [6.15119272, 2.82397981, 4.24282686, 1.75378872, 4.80532629, 3.16535161],
                    [2.7196025, 2.17993876, 2.79799651, 7.20950623, 4.70827355, 2.42446381],
                    [0.29748325, 8.22965311, 0.07526586, 1.98395573, 1.77853129, 5.00793316],
                    [6.37849798, 3.80507597, 2.5126212, 0.75632265, 2.49531244, 5.63243171],
                    [0.79285198, 4.00586269, 0.36314749, 8.9344773, 1.82041716, 0.2318847],
                    [0.24871352, 3.25946831, 3.9988045, 6.9335196, 4.81556096, 1.43535682],
                    [5.2736312, 0.59346769, 0.73640014, 7.30730989, 4.09948515, 1.0448773],
                    [1.74241088, 2.32320373, 9.17490044, 2.28211094, 1.47515927, 0.06168781],
                    [1.65116829, 3.72063198, 5.63953167, 0.25461896, 6.35720791, 3.33875729],
                    [2.5078766, 4.59291179, 0.81935207, 8.24752456, 0.33308447, 1.95237595],
                    [1.05128312, 4.85979168, 3.28552824, 6.26921471, 3.39863537, 3.69171469],
                    [6.30499955, 1.82204004, 1.93686289, 3.35062427, 1.83174219, 6.21238686],
                    [4.74718378, 6.36499948, 4.05818821, 4.43996757, 0.42190953, 0.76864591],
                    [1.25720612, 0.74301296, 1.3374366, 8.30597947, 5.08394071, 1.1148452],
                    [0.63888729, 0.28507461, 4.87857435, 6.41971655, 5.85711844, 0.43757381],
                    [0.74870183, 2.51804488, 6.59949427, 2.14794505, 6.05084902, 2.88429005],
                    [3.57753129, 3.67307393, 5.43392619, 2.06131042, 2.63388133, 5.74420686],
                    [3.94583726, 0.62586462, 0.72667245, 9.06686254, 1.13056724, 0.15630224],
                    [2.53054533, 4.2406129, 2.22057705, 7.51774642, 3.47885032, 1.43654771],
                    [1.63510684, 3.25906419, 0.37991887, 7.02694214, 2.53469812, 5.54598751],
                    [7.11491625, 1.26647073, 5.01203819, 4.52740681, 1.16148237, 0.89835304],
                    [2.75824608, 5.28476545, 2.49891273, 0.63079997, 7.07433925, 2.78829399],
                    [4.92392025, 4.74424707, 2.56041791, 4.76935788, 1.43523334, 4.67811073],
                    [2.43924518, 1.00523211, 6.09587506, 1.47285316, 6.69893956, 2.972341],
                    [1.14431283, 4.55594834, 4.12473926, 5.80221944, 1.92147095, 4.85413307],
                    [7.08401121, 1.66591657, 2.90546299, 2.62634248, 3.62934098, 4.30464879],
                    [0.71623214, 3.11241519, 1.7018771, 7.50296641, 5.38823009, 1.25537605],
                    [1.33651336, 4.76969307, 0.64008086, 6.48262472, 5.64538051, 1.07671362],
                    [3.09497945, 1.2275849, 3.84351994, 7.19938601, 3.78799616, 2.82159852],
                    [5.06781785, 3.12557557, 6.88555034, 1.21769126, 2.73086695, 2.86300362],
                    [8.30192712, 0.40973443, 1.69099424, 4.54961192, 2.64473811, 0.59753994],
                    [5.96294481, 6.46817991, 1.35988062, 2.83106174, 0.74946184, 3.48999411],
                    [0.43320751, 1.24640954, 5.6313907, 1.62670791, 4.58871327, 6.54551489],
                    [3.7064827, 7.60850058, 3.73003227, 2.71892257, 1.4363049, 2.23697394],
                    [4.44128859, 1.8202686, 4.22272069, 2.30194565, 0.67272146, 7.30607281],
                    [0.93689572, 0.77924846, 2.83896436, 1.98294555, 8.45958836, 3.86763124],
                    [1.12281975, 2.73059913, 0.32294675, 2.84237021, 1.68312155, 8.95917647],
                    [4.27687318, 2.83055698, 5.27541783, 5.03273808, 0.01475194, 4.53184284],
                    [3.73578206, 6.07088863, 2.17391882, 4.89911933, 0.27124696, 4.51523815],
                    [6.05671623, 0.7444296, 4.30057711, 3.09050824, 1.16194731, 5.77630391],
                    [1.40468169, 5.19102545, 6.72110624, 4.75666122, 0.91486715, 1.56334486],
                    [4.41604152, 0.86551038, 2.05709774, 4.70986355, 3.106477, 6.60944809],
                    [5.95498781, 5.94146861, 4.17018388, 0.93397018, 0.89950814, 3.18829456],
                    [9.59164585, 1.48925818, 0.72278285, 2.04850964, 1.0181982, 0.16402902],
                    [4.4579775, 3.16479945, 1.00362159, 2.24428595, 7.91409455, 1.19729395],
                    [2.12268361, 0.64607954, 6.43093367, 0.73854263, 6.94484318, 2.22341982],
                    [3.08973572, 5.6223787, 0.9737901, 5.75218769, 3.94430958, 3.04119754],
                    [2.5850297, 0.26144699, 2.28343938, 8.50777354, 3.93535625, 0.40734769],
                    [4.72502594, 5.38532887, 5.40386645, 1.57883722, 0.24912224, 4.11288237]],
                  '5':
                   [[ 3.67917966,  0.38835143,  8.09989551,  2.86026356,  3.24527031,  1.41124976],
                    [ 7.49190652,  0.86177565,  0.26446419,  6.40116659,  1.13497678,  0.89198211],
                    [ 3.14072363,  8.1320309 ,  3.56036928,  2.95551047,  0.38337821,  1.56450569],
                    [ 0.03085158,  4.25364725,  3.34139266,  4.67838906,  1.98970378,  6.70032708],
                    [ 4.02109647,  4.65093134,  5.52044309,  0.41989912,  5.07013412,  2.41697202],
                    [ 2.96104264,  6.42797292,  4.00884559,  2.28915409,  0.82767172,  5.28368061],
                    [ 3.95849765,  4.90714693,  3.91729584,  2.69024104,  6.08226306,  0.82077889],
                    [ 0.74185053,  1.02527749,  5.89640728,  5.80289307,  2.44397849,  4.89737136],
                    [ 4.2850684 ,  0.09305206,  7.94851261,  1.77192616,  2.93208106,  2.59111093],
                    [ 5.93102382,  4.12666154,  0.77446586,  4.31927672,  5.33551751,  0.26443358],
                    [ 1.05416268,  2.2897475 ,  1.46517302,  2.79084328,  9.0996314 ,  0.95234852],
                    [ 0.79064992,  0.57247091,  4.45310153,  6.54823417,  6.00440567,  0.53364626],
                    [ 4.35145246,  0.63842407,  0.7493827 ,  1.11659248,  7.71912589,  4.38907945],
                    [ 2.44514113,  3.33980397,  2.63961606,  1.86832856,  8.08225336,  2.66194486],
                    [ 6.66924742,  3.12365865,  5.35457136,  1.0395009 ,  0.1690967 ,  3.99791261],
                    [ 4.45750971,  3.52046079,  3.30278422,  3.47142192,  4.12997745,  5.26508267],
                    [ 1.70141153,  6.84591866,  0.69367176,  1.01722498,  6.77031075,  1.69869412],
                    [ 1.61886242,  1.17208561,  0.89685985,  3.80015468,  8.79718107,  1.83563934],
                    [ 2.36047279,  2.6207669 ,  7.50292079,  0.53373023,  5.49537505,  0.88425889],
                    [ 0.52087376,  3.04167888,  0.13469428,  7.33280189,  2.69762123,  5.42324568],
                    [ 4.85800708,  1.07551152,  0.44983357,  3.87424429,  5.7594417 ,  5.18263972],
                    [ 3.51780254,  1.44977209,  2.26423569,  6.76102431,  5.18387135,  2.79508338],
                    [ 0.92487495,  0.88767274,  2.25049907,  2.94454564,  2.46602119,  8.86229586],
                    [ 0.91807423,  2.21441185,  3.17003378,  8.38445357,  1.6293076 ,  3.35420641],
                    [ 1.00725144,  3.13922254,  5.11333093,  2.46965818,  6.66930986,  3.52216804],
                    [ 1.89889027,  4.26005076,  3.8055043 ,  2.601719  ,  2.73103432,  7.03824055],
                    [ 4.57737545,  1.06276164,  4.19723485,  6.86155259,  2.25429762,  2.85282836],
                    [ 1.23346934,  1.23694538,  9.64358056,  0.16691509,  0.29485875,  1.95833379],
                    [ 1.51947586,  8.43245754,  4.84809716,  0.58788756,  0.37534243,  1.61068716],
                    [ 3.8159377 ,  2.93304432,  0.41425422,  5.33636409,  1.0104518 ,  6.8677849 ],
                    [ 5.85626898,  4.80785389,  1.58310642,  3.48908773,  2.29308951,  4.7592474 ],
                    [ 4.24912001,  6.97078189,  4.74298955,  1.38027302,  0.67165641,  2.91563942]],
                  '7':
                   [[  9.49729374,  2.98910393,  0.19374418,  0.48817863,   0.75034508,  0.16672279],  
                    [  1.74327056,  0.46482846,  7.55950543,  1.57177559,   5.29791865,  3.01004973],  
                    [  7.44433577,  2.27422887,  4.78271726,  3.46265870,   0.50993921,  2.07010153],  
                    [  4.78968371,  3.36538593,  0.88592964,  6.38949462,   2.00825232,  4.48213313],  
                    [  5.77532710,  5.31652260,  2.38513502,  2.08315748,   4.17604119,  3.30339978],  
                    [  6.88088248,  6.02843423,  0.02104655,  4.00481529,   0.48111127,  0.20243633],  
                    [  0.96132316,  1.91371320,  0.94262497,  3.51414795,   1.61855745,  8.91942003],  
                    [  2.06562582,  5.11098416,  6.29032188,  4.60098502,   0.38568959,  2.95382160],  
                    [  0.58513677,  4.00782651,  0.08530382,  8.62832436,   1.93453438,  2.32320047],  
                    [  1.24387716,  2.36413212,  3.54918392,  7.29005232,   5.20695449,  0.09851170],  
                    [  4.42188476,  1.93117784,  7.33958506,  4.19677527,   0.94119320,  2.08547622],  
                    [  2.22509178,  5.69526240,  4.02537895,  5.51223418,   3.90230246,  0.89251721],  
                    [  5.63262741,  0.25607217,  4.21983956,  5.34274324,   4.06989729,  2.30041742],  
                    [  3.99576756,  3.36409888,  5.49561503,  3.48562801,   5.14196367,  1.98128815],  
                    [  3.86901361,  2.35718170,  0.87556104,  0.82003307,   8.78839182,  0.89416779],  
                    [  0.39341137,  4.84127331,  1.70702545,  1.67922015,   8.35107069,  0.96602394],  
                    [  3.45410722,  7.40358442,  4.80502365,  1.81341621,   0.66537847,  2.53704984],  
                    [  5.10549280,  6.36812455,  4.58494717,  1.22495898,   2.39242255,  2.26604992],  
                    [  2.10660119,  0.49899214,  2.13559315,  0.77146133,   1.82738095,  9.31761807],  
                    [  2.41451522,  6.11407574,  0.02657181,  3.15297076,   6.55980963,  1.95324369],  
                    [  5.07301207,  1.04764679,  0.38126799,  1.75203503,   4.83320736,  6.82584056],  
                    [  5.86496023,  5.29477535,  0.49074425,  5.02714001,   1.04921431,  3.30964927],  
                    [  6.04985605,  2.15313597,  3.01872538,  2.14930505,   6.31431071,  2.27167613],  
                    [  3.38761354,  1.73456127,  2.59356744,  4.78587994,   7.35331757,  1.34642252],  
                    [  4.61573836,  0.10237836,  3.84736614,  7.75293952,   1.19594946,  1.53097533],  
                    [  6.49637377,  5.04239913,  0.20827311,  3.29038975,   0.33107957,  4.62511436],  
                    [  2.30298197,  3.29946741,  4.49935456,  5.34601600,   5.21029158,  2.79974498],  
                    [  0.54574584,  2.42444606,  1.60434868,  3.73649348,   1.26056229,  8.70056821],  
                    [  3.61195169,  6.65026280,  3.74800770,  2.29512444,   0.09778884,  4.83767393],  
                    [  7.63538382,  3.88682729,  3.22292045,  0.21632413,   0.07119400,  4.01925448],  
                    [  2.96169175,  0.69908964,  1.91478181,  3.19293209,   8.58251758,  1.79411341],  
                    [  2.90919565,  5.51711193,  1.75245139,  2.91477850,   6.98631598,  0.84995649],  
                    [  6.76482657,  5.05071646,  3.62871840,  2.62715174,   1.66996898,  2.42261528],  
                    [  8.03143521,  2.95933932,  3.39642327,  3.24934646,   2.10962576,  0.44033503],  
                    [  1.80722860,  3.77826266,  0.16093320,  0.03592059,   4.83503526,  7.68465356],  
                    [  7.22319621,  0.66064721,  1.26699526,  5.78540374,   2.55433573,  2.40586313],  
                    [  4.03459556,  0.68212834,  1.37929110,  8.75546017,   1.76414640,  1.25857076],  
                    [  5.73690431,  1.64965556,  0.64870873,  5.25548535,   5.84541730,  1.46857511],  
                    [  2.53210586,  3.05479350,  0.38493697,  7.20837806,   5.66222550,  0.29493761],  
                    [  3.89471153,  5.32895093,  1.35814442,  4.04568659,   4.12513338,  4.60484989],  
                    [  2.41451158,  1.50862356,  1.41410605,  0.51982294,   5.57708595,  7.64986205],  
                    [  5.34003191,  1.89167534,  2.58671856,  0.85251419,   6.32481989,  4.52596770],  
                    [  0.57978779,  6.09839275,  7.76016689,  0.41757963,   0.93387774,  1.09852695],  
                    [  0.00888177,  0.18279964,  6.91394786,  2.79976789,   1.50722126,  6.48486038],  
                    [  1.19433531,  6.29760769,  1.55051483,  3.59191723,   6.60212631,  0.14022512],  
                    [  4.67099335,  3.31383981,  1.63220871,  2.23502881,   6.12885063,  4.68807186],  
                    [  7.30408120,  3.80874526,  3.72204577,  0.66679044,   3.76398221,  1.91782718],  
                    [  5.92805521,  2.31772110,  2.46175229,  4.46357240,   1.63871164,  5.55132881],  
                    [  1.68150470,  2.41378962,  3.08030571,  2.26997261,   6.68549178,  5.65767640],  
                    [  5.91934684,  1.69803498,  2.00819247,  0.77198553,   7.45425308,  1.37234202],  
                    [  5.53172263,  1.58718247,  2.90049338,  3.84536120,   4.93451202,  4.39679691],  
                    [  1.81587798,  3.88103491,  6.91095741,  2.44678365,   1.42743937,  5.08473102],  
                    [  2.55234884,  1.33311404,  4.82796793,  6.60348354,   4.37598182,  2.37567598],  
                    [  5.12272256,  2.03525835,  0.58747523,  7.29581125,   3.77362349,  1.34209305],  
                    [  0.27055161,  1.32505827,  0.82880694,  5.07516063,   3.65750329,  7.63868547],  
                    [  1.46014514,  1.17338868,  1.66221135,  3.56456484,   7.12719319,  5.49774348],  
                    [  3.98703736,  1.50123650,  3.76330075,  8.02615543,   1.75580291,  0.43055159],  
                    [  0.60157514,  1.24011244,  3.12416622,  3.63995686,   4.51357407,  7.39717359],  
                    [  3.12439557,  8.26536950,  0.88929913,  4.26851883,   1.38154049,  1.00102909],  
                    [  7.26979727,  2.40760022,  4.84651024,  3.36817790,   1.75647764,  1.85337834],  
                    [  5.29377498,  0.84307824,  6.79950126,  3.34597688,   0.88493918,  3.61293088],  
                    [  4.63626603,  0.39739091,  0.76906016,  4.12608336,   1.14102927,  7.70903059],  
                    [  7.30424808,  1.14559469,  2.98920677,  3.67674168,   3.92079189,  2.74028780],  
                    [  5.29547651,  2.85826185,  6.93657406,  1.33672260,   3.70102745,  0.43333184],  
                    [  3.49313310,  4.54336493,  1.56445202,  4.02157354,   6.96400202,  0.19485449],  
                    [  2.73742580,  3.81123532,  5.00271032,  1.84885547,   3.60970535,  6.04198936],  
                    [  1.31467477,  2.17922309,  2.63299757,  0.16194893,   9.27658681,  0.71319734],  
                    [  2.76492957,  4.10249912,  5.39611053,  5.09161003,   3.89378732,  2.30663915],  
                    [  1.94582268,  1.87031650,  4.23230029,  8.24909730,   1.84242598,  1.83335339],  
                    [  3.39276969,  4.78530893,  1.24380350,  7.20460307,   1.71752335,  3.03095584],  
                    [  7.75709126,  2.86264536,  2.29374145,  2.68872195,   3.44058744,  2.70271701],  
                    [  1.47805473,  0.56765227,  7.37702891,  0.09271143,   4.36384564,  4.90110451],  
                    [  1.82149711,  2.26115095,  1.21455515,  6.81753044,   3.70962435,  5.46389662],  
                    [  2.36459101,  0.12105503,  0.24822624,  3.62858762,   8.70938165,  2.30487785],  
                    [  0.78787403,  3.72625502,  1.21190521,  8.38489292,   3.70155394,  0.13278932],  
                    [  0.75428354,  7.27153941,  4.72464149,  3.80635658,   0.68907238,  3.04472702],  
                    [  5.12459705,  2.87302371,  0.43644150,  2.27226903,   7.07471522,  3.17473726],  
                    [  0.70178603,  2.88216063,  4.06577522,  6.39365228,   5.80793442,  0.24336517],  
                    [  6.99582269,  0.77356834,  2.80788891,  0.88674480,   2.35952093,  6.01848497],  
                    [  3.69693603,  0.85342739,  8.50506298,  1.79107374,   2.34725025,  2.13323710],  
                    [  3.09834673,  3.77166522,  3.14744542,  8.10858723,   0.17928020,  0.69788831],  
                    [  3.26246217,  3.77555064,  5.13658315,  5.95335244,   0.50710019,  3.60797944],  
                    [  5.71260164,  2.65661250,  0.79608313,  0.61669039,   1.68623679,  7.51339754],  
                    [  1.73776892,  2.79515779,  5.26232705,  6.15519998,   2.91607776,  3.88396316],  
                    [  5.84288335,  2.11962167,  3.05793046,  0.02686868,   2.23735440,  6.85642056],  
                    [  3.91342565,  1.03017066,  1.54153396,  9.00377711,   0.12064941,  0.40615586],  
                    [  3.43153396,  0.09349497,  5.55042223,  0.76858599,   6.17367356,  4.32477477],  
                    [  2.77747993,  1.37010370,  7.76474417,  0.73141948,   5.35935029,  0.92712408],  
                    [  0.79943355,  8.13116642,  3.40436523,  0.56073057,   4.53168239,  0.89709087],  
                    [  1.86322992,  3.54240501,  4.19454516,  2.38460530,   4.28548294,  6.50644492],  
                    [  2.28255306,  2.45104299,  0.21768383,  1.84082889,   6.48114612,  6.58339180],  
                    [  3.76567476,  2.57286633,  5.15499273,  6.27845094,   1.49352325,  3.31308685],  
                    [  0.95794499,  1.73192724,  5.52153846,  3.27936890,   7.20088206,  1.72870459],  
                    [  5.26466955,  6.16182728,  5.64238349,  0.09414682,   0.83475936,  1.33152576],  
                    [  6.93893651,  5.88859230,  0.31336892,  2.38592572,   0.64748960,  3.31142816],  
                    [  3.12497486,  3.76754998,  4.92020336,  3.59103512,   2.40286990,  5.75867877],  
                    [  1.84234679,  2.60493608,  4.71846310,  0.96653102,   8.14264373,  0.56510457],  
                    [  4.90541585,  6.68462108,  3.29352326,  1.25075319,   4.14418724,  1.29103422],  
                    [  0.77144415,  4.40756648,  2.83853600,  3.65761195,   7.28753539,  2.33123310],  
                    [  6.65232787,  0.61323121,  0.05237419,  3.02035801,   4.15410725,  5.38410334],  
                    [  5.31726305,  2.11795941,  2.65122017,  1.55577325,   7.31211902,  2.07953799],  
                    [  6.80965923,  2.23215998,  2.63179708,  6.09494365,   1.63349762,  1.37949068],  
                    [  2.87351031,  0.55340637,  5.41604829,  3.55340001,   1.91207540,  6.76907800],  
                    [  4.39372404,  0.06868095,  4.42865756,  5.35634714,   1.48055098,  5.49499576],  
                    [  1.04344792,  0.31173723,  1.56573663,  5.79359716,   2.79947656,  7.41347881],  
                    [  6.79414017,  0.09388667,  6.72878775,  2.10117159,   1.75210434,  1.03415104],  
                    [  1.08440052,  5.95542225,  0.39378945,  2.98381656,   3.56511194,  6.44893532],  
                    [  0.99064629,  1.35478691,  4.38745525,  2.66407385,   7.56658676,  3.68549645],  
                    [  7.14046972,  4.96484612,  3.19339033,  1.54451680,   3.42381000,  0.24134276],  
                    [  8.79248611,  2.96251195,  2.89483138,  1.82850443,   0.12129745,  1.47564250],  
                    [  2.94607977,  5.56169236,  2.59013018,  4.02632765,   3.30995993,  5.14900658],  
                    [  0.88402512,  0.35402459,  3.25443105,  6.37560825,   6.50357815,  2.35731528],  
                    [  1.94480257,  1.35885649,  0.88307848,  0.22492469,   3.33476206,  9.07855690],  
                    [  2.48584079,  5.07531399,  4.43407763,  4.51885124,   3.70528802,  3.77512426],  
                    [  7.48552913,  3.84871747,  2.91583698,  3.06351150,   0.02403987,  3.35655202],  
                    [  3.59720211,  4.94644652,  2.96841414,  4.91597513,   4.88131902,  2.40566715],  
                    [  1.87916271,  8.77986293,  0.14218332,  2.30457011,   2.26344244,  2.98803000],  
                    [  2.83343404,  4.99152641,  4.61233030,  6.57058571,   0.84136667,  1.37921369],  
                    [  0.95627524,  7.31066478,  4.38424188,  3.81744136,   3.40120183,  0.52641131],  
                    [  2.37882278,  7.66799820,  2.56820049,  3.40968706,   3.97975549,  1.21775717],  
                    [  3.90209510,  1.06132813,  0.91070059,  3.86200052,   1.17124110,  8.15665416],  
                    [  4.61696992,  4.62523330,  5.02800260,  4.30399729,   3.24034661,  1.72793421],  
                    [  2.21521553,  1.61363547,  2.97780427,  7.14111649,   1.51642660,  5.50695815],  
                    [  5.65137310,  0.06657199,  0.99745488,  4.47419538,   3.55469208,  5.86586515],  
                    [  0.68558487,  0.30887798,  2.04370625,  9.13806017,   3.13470834,  1.38826961],  
                    [  2.61242526,  6.59409851,  4.45452192,  3.66950713,   4.03746871,  0.28965048],  
                    [  7.79480886,  4.68269928,  3.85253341,  0.20850008,   1.55792871,  0.02558407],  
                    [  1.68967122,  1.11253309,  3.74425011,  3.12606095,   3.20780397,  7.86292624]]
                }
        FRUITS = np.array(FRUITS_DICT[str(args.depth)]) * np.power(args.gamma, args.depth-1)
        return FRUITS
    
def find_in_dst(A, B, base=0):
    # base = 0: tolerance w.r.t. A
    # base = 1: tolerance w.r.t. B
    # base = 2: no tolerance
    cnt = 0.0
    for a in A:
        for b in B:
            if base == 0:
              if np.linalg.norm(a - b, ord=1) < 0.20*np.linalg.norm(a):
                  cnt += 1.0
                  break
            elif base == 1:
              if np.linalg.norm(a - b, ord=1) < 0.20*np.linalg.norm(b):
                  cnt += 1.0
                  break
            elif base == 2:
              if np.linalg.norm(a - b, ord=1) < 0.3:
                  cnt += 1.0
                  break
    return cnt / len(A)

def find_in_ftn(A, B, eps=0.2):
    # find element of A in B with a tolerance of relative err of eps.
    cnt1, cnt2 = 0.0, 0.0
    for a in A:
        for b in B:
            if eps > 0.001:
              if np.linalg.norm(a - b, ord=1) < eps*np.linalg.norm(b):
                  cnt1 += 1.0
                  break
            else:
              if np.linalg.norm(a - b, ord=1) < 0.5:
                  cnt1 += 1.0
                  break
    for b in B:
        for a in A:
            if eps > 0.001:
              if np.linalg.norm(a - b, ord=1) < eps*np.linalg.norm(b):
                  cnt2 += 1.0
                  break
            else:
              if np.linalg.norm(a - b, ord=1) < 0.5:
                  cnt2 += 1.0
                  break
    return cnt1, cnt2

def compute_sparsity(obj_batch):
    non_dom = NonDominatedSorting().do(obj_batch, only_non_dominated_front=True)        
    objs = obj_batch[non_dom]
    sparsity_sum = 0     
    for objective in range(objs.shape[-1]):
        objs_sort = np.sort(objs[:,objective])
        sp = 0
        for i in range(len(objs_sort)-1):
            sp +=  np.power(objs_sort[i] - objs_sort[i+1],2)
        sparsity_sum += sp
    if len(objs) > 1:
        sparsity = sparsity_sum/(len(objs)-1)
    else:   
        sparsity = 0
    
    
    return sparsity


def generate_w_batch_test(args, step_size):
    mesh_array = []
    step_size = step_size
    for i in range(args.reward_size):
        mesh_array.append(np.arange(0,1+step_size, step_size))
        
    w_batch_test = np.array(list(itertools.product(*mesh_array)))
    w_batch_test = w_batch_test[w_batch_test.sum(axis=1) == 1,:]
    w_batch_test = np.unique(w_batch_test,axis =0)
    
    return w_batch_test

def plot_objs(args,objs,figure_dir, ext=''):
    non_dom = NonDominatedSorting().do(-objs, only_non_dominated_front=True)        
    objs_plot = objs[non_dom]
    plt.plot(objs_plot[:,0],objs_plot[:,1],'rs', markersize=2)
    plt.xlabel("Speed")
    plt.ylabel("Energy Efficiency")
    plt.title('{} - Pareto Front'.format(args.scenario_name))
    if args.scenario_name == "MO-Walker2d-v2":
        plt.ylim(-100, 2700)
        plt.xlim(-100, 2700)
    elif args.scenario_name == "MO-HalfCheetah-v2":
        plt.ylim(0, 2700)
        plt.xlim(0, 2700)
    elif args.scenario_name == "MO-Ant-v2":
        plt.ylim(0, 3700)
        plt.xlim(0, 3400)
    elif args.scenario_name == "MO-Swimmer-v2":
        plt.ylim(0, 300)
        plt.xlim(0, 300)
    elif args.scenario_name == "MO-Hopper-v2":
        plt.ylim(0, 6000)
        plt.xlim(0, 5000)
    plt.savefig('{}/{}-ParetoFront, ExpNoise={}, PolicyUpdateFreq={}_{}.png'.format(figure_dir, args.plot_name,args.expl_noise,args.policy_freq,ext))
    numpy_filename = '{}/{}-ParetoFront, ExpNoise={}, PolicyUpdateFreq={}_{}.npy'.format(figure_dir,args.plot_name,args.expl_noise,args.policy_freq,ext)
    with open(numpy_filename, 'wb') as f:
        np.save(f, objs)
    plt.close()
    

# Evaluate agent druing training 
def eval_agent(test_env, agent, w_batch, args, eval_episodes=1):
    
    use_cuda = args.cuda
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    hypervolume,  sparsity = np.zeros((eval_episodes,)), np.zeros((eval_episodes,))
    objs = np.zeros((eval_episodes,len(w_batch),args.reward_size))
    for eval_ep in range(eval_episodes):
        test_env.seed(eval_ep*11)
        test_env.action_space.seed(eval_ep*11)
        torch.manual_seed(eval_ep*11)
        np.random.seed(eval_ep*11)    
        
        recovered_objs = []
        # Evaluate agent for the preferences in w_batch
        for evalPreference in w_batch:
            evalPreference = np.abs(evalPreference) / np.linalg.norm(evalPreference, ord=1, axis=0, keepdims=True)
            evalPreference = FloatTensor(evalPreference)
            # reset the environment
            state = test_env.reset()
            terminal = False
            tot_rewards = 0
            cnt = 0
            # interact with the environment
            while not terminal:
                if hasattr(agent, 'deterministic'):
                    action = agent(state, evalPreference, deterministic = True)
                else:
                    action = agent(state, evalPreference)
                next_state, reward, terminal, _ = test_env.step(action)
                tot_rewards += reward 
                state = next_state
                cnt +=1
            recovered_objs.append(tot_rewards)
            if hasattr(agent, 'deterministic'): 
                agent.reset_preference()
        recovered_objs = np.array(recovered_objs)
        # Compute hypervolume and sparsity
        perf_ind = get_performance_indicator("hv", ref_point = np.zeros((recovered_objs.shape[1])))
        hv = perf_ind.do(-recovered_objs)
        s = compute_sparsity(-recovered_objs)
        hypervolume[eval_ep] = hv
        sparsity[eval_ep] = s
        objs[eval_ep] = recovered_objs
    return hypervolume.mean(), sparsity.mean(), objs.mean(axis=0)
	
# Evaluate agent after training
def eval_agent_test(test_env, agent, w_batch, args, eval_episodes=1):
    
    use_cuda = args.cuda
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    hypervolume,  sparsity = np.zeros((eval_episodes,)), np.zeros((eval_episodes,))
    objs = np.zeros((eval_episodes,len(w_batch),args.reward_size))
    for eval_ep in range(eval_episodes):
        test_env.seed(eval_ep*11)
        test_env.action_space.seed(eval_ep*11)
        torch.manual_seed(eval_ep*11)
        np.random.seed(eval_ep*11)    
        
        # compute recovered Pareto
        recovered_objs = []
        pref_cnt = 0
        for evalPreference in w_batch:
            pref_cnt += 1
            evalPreference = np.abs(evalPreference) / np.linalg.norm(evalPreference, ord=1, axis=0, keepdims=True)
            evalPreference = FloatTensor(evalPreference)
            # reset the environment
            state = test_env.reset()
            terminal = False
            tot_rewards = 0
            # interact with the environment
            while not terminal:
                # state = env.observe()
                if hasattr(agent, 'deterministic'):
                    action = agent(state, evalPreference, deterministic = True)
                else:
                    action = agent(state, evalPreference)
                next_state, reward, terminal, _ = test_env.step(action)
                tot_rewards += reward 
                state = next_state
            recovered_objs.append(tot_rewards)
            if hasattr(agent, 'deterministic'): 
                agent.reset_preference()
            print("Preference %d out of %d Preferences" % (pref_cnt, len(w_batch)))
        # Compute hypervolume and sparsity
        recovered_objs = np.array(recovered_objs)
        perf_ind = get_performance_indicator("hv", ref_point = np.zeros((recovered_objs.shape[1])))
        hv = perf_ind.do(-recovered_objs)
        s = compute_sparsity(-recovered_objs)
        hypervolume[eval_ep] = hv
        sparsity[eval_ep] = s
        objs[eval_ep] = recovered_objs
    return hypervolume, sparsity, objs

# Evaluation for interpolator update
def eval_agent_interp(test_env, agent, w_batch, args, eval_episodes=1):
    
    use_cuda = args.cuda
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    objs = np.zeros((eval_episodes,len(w_batch),args.reward_size))
    for eval_ep in range(eval_episodes):
        test_env.seed(eval_ep*11)
        test_env.action_space.seed(eval_ep*11)
        torch.manual_seed(eval_ep*11)
        np.random.seed(eval_ep*11)    
        
        # compute recovered Pareto
        recovered_objs = []
        avg_preference = 0
        for evalPreference in w_batch:
            evalPreference = np.abs(evalPreference) / np.linalg.norm(evalPreference, ord=1, axis=0, keepdims=True)
            evalPreference = FloatTensor(evalPreference)
            # reset the environment
            state = test_env.reset()
            terminal = False
            tot_rewards = 0
            cnt = 0
            # interact with the environment
            while not terminal:
                if hasattr(agent, 'deterministic'):
                    action = agent(state, evalPreference, deterministic = True)
                else:
                    action = agent(state, evalPreference)
                next_state, reward, terminal, _ = test_env.step(action)
                tot_rewards += reward 
                state = next_state
                cnt +=1
            recovered_objs.append(tot_rewards)
            avg_preference += evalPreference
            if hasattr(agent, 'deterministic'): 
                agent.reset_preference()
        recovered_objs = np.array(recovered_objs)
        objs[eval_ep] = recovered_objs
    return objs.mean(axis=0)

def eval_agent_discrete(env, agent, w_batch, args, metric_calc=True):
    
    use_cuda = args.cuda
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    
    # compute recovered Pareto
    recovered_objs = []
    with torch.no_grad():
        for evalPreference in w_batch:
            evalPreference = np.abs(evalPreference) / np.linalg.norm(evalPreference, ord=1, axis=0, keepdims=True)
            evalPreference = FloatTensor(evalPreference)
            # reset the environment
            state = env.reset()
            terminal = False
            cnt = 0
            tot_rewards = 0
            # interact with the environment
            while not terminal:
                # state = env.observe()
                if hasattr(agent, 'deterministic'):
                    action = agent(state, evalPreference, deterministic = True)
                else:
                    action = agent(state, evalPreference)
                next_state, reward, terminal, _ = env.step(action)
                tot_rewards += reward * np.power(args.gamma, cnt)
                cnt += 1
                if cnt > args.max_episode_len:
                    terminal = True
                state = next_state
            recovered_objs.append(tot_rewards)
        if hasattr(agent, 'deterministic'): 
            agent.reset_preference()
        #Obtain coverage ratio f1 score for DeepSeaTreasure
        if args.scenario_name == 'dst-v0':
            time, treasure = real_pareto_cal(args) # Compute actual pareto curve
            objs = np.vstack(recovered_objs)
            true_obj = np.vstack((treasure,time))
            true_obj = true_obj.transpose()
            # Compute precision: Ratio of optimal solutions recovered among the total recovered solutions
            objs_precision = find_in_dst(objs, true_obj, 2) 
            # Compute recall: Ratio of optimal solutions recovered to the total optimal solutions
            objs_recall = find_in_dst(true_obj, objs, 2)
            # Compute coverage ratio, agent's ability to recover optimal solutions
            try:
                CR_F1 = 2 * objs_precision * objs_recall / (objs_precision + objs_recall) 
            except ZeroDivisionError:
                CR_F1 = 0
            objs_tmp = np.zeros(objs.shape)
            objs_tmp[:,1] = 19
            objs_tmp = objs_tmp + objs

            # Compute hypervolume and sparsity
            hv = get_performance_indicator("hv", ref_point = np.array([0, 0]))
            hypervolume = hv.do(-objs_tmp)
            sparsity = compute_sparsity(-objs_tmp)

            return CR_F1, hypervolume, sparsity, objs
        #Obtain coverage ratio f1 score for FruitTaskNavigation
        if args.scenario_name == 'ftn-v0':
            true_obj = real_pareto_cal(args) # Compute actual pareto curve
            objs = np.array(recovered_objs)
            cnt1, cnt2 = find_in_ftn(objs, true_obj, 0.0)
            # Compute precision: Ratio of optimal solutions recovered among the total recovered solutions
            objs_precision = cnt1 / len(objs)
            # Compute recall: Ratio of optimal solutions-recovered to the total optimal solutions
            objs_recall = cnt2 / len(true_obj)
            # Compute coverage ratio, agent's ability to recover optimal solutions
            try:
                CR_F1 = 2 * objs_precision * objs_recall / (objs_precision + objs_recall) 
            except ZeroDivisionError:
                CR_F1 = 0           
            # Compute hypervolume and sparsity    
            if metric_calc == True:
                hv = get_performance_indicator("hv", ref_point = np.zeros((objs.shape[1])))
                hypervolume = hv.do(-objs)
                sparsity = compute_sparsity(-objs)
            else:
                sparsity = 0
                hypervolume = 0
            return CR_F1, hypervolume, sparsity, objs
    

def store_results(CR_F1, hypervolume, sparsity, time_metric, writer, args):
        if args.scenario_name == "MO-Walker2d-v2" or args.scenario_name == "MO-HalfCheetah-v2" or args.scenario_name == "MO-Swimmer-v2" or args.scenario_name == "MO-Ant-v2"  or args.scenario_name == "MO-Hopper-v2":
            print("Time Step %d Hypervolume %0.2f; Sparsity %0.2f" % (time_metric, hypervolume, sparsity))
            writer.add_scalar("Evaluation Metric/Hypervolume", hypervolume, time_metric)
            writer.add_scalar("Evaluation Metric/Sparsity", sparsity, time_metric)
        else:
            print("Time Step %d with CR_F1 %0.2f; Hypervolume %0.2f; Sparsity %0.2f" % (time_metric, CR_F1, hypervolume, sparsity))
            writer.add_scalar("Evaluation Metric/CRF1", CR_F1, time_metric)
            writer.add_scalar("Evaluation Metric/Hypervolume", hypervolume, time_metric)
            writer.add_scalar("Evaluation Metric/Sparsity", sparsity, time_metric)