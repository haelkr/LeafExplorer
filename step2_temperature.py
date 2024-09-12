# -*- coding: utf-8 -*-

#to add: Ausgabe wenn nichts erkannt wird

"""
Store the other two image types also in seperate directories

Guide:
    Set ROOT_DIR and DIR paths, upload the images to analyze (rgb, thermal image and csv file) into the corresponding directories 
        IMAGE_DIR = os.path.join(DIR, "preprocessed_data", "RGB")
        THERMAL_DIR = os.path.join(DIR, "preprocessed_data", "csv")
        THERMAL_IMG_DIR = os.path.join(DIR, "preprocessed_data", "thermal")
    Set WEIGHTS_PATH= os.path.join(MODEL_DIR , "default", "mask_rcnn_object_0100.h5")
    
    The model detects instances on the images in IMAGE_DIR (classification, object detection, masking)
    Creates an image of all detected leaves, determies the leaf centers
    Determines the 6 closest leaves to the plant center
    Analyzes the csv file for those leaves:
        max, min, average T and size for every leaf
        calculate mean value for all 6, store it in the final_results csv file in RESULTS_DIR
        stores the histogram of the 6 leaves in seperate folders in HIST_DIR
    Stores the mask of all leaves and only the 6 closest leaves in MASK_DIR
    In RESULTS_DIR there is also the image with all masks where the 6 leaf center are indicated as well as the plant center
"""
print("importing packages...")
#import warnings
 

import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import cv2, os,csv, re
#import random, math,glob
import skimage, json

import mrcnn
#from mrcnn2.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import mrcnn.visualize as visualize

from PIL import Image, ImageDraw
from tqdm import tqdm
from plantcv import plantcv as pcv
import functions as fc

import gc

from mrcnn_colab_engine_needed_for_saqib_detection import draw_mask, detect_contours_maskrcnn_with_original#,random_colors,   detect_contours_maskrcnn,get_mask_contours

# =============================================================================
# from collections import defaultdict
# from gc import get_objects
# before = defaultdict(int)
# after = defaultdict(int)
# for i in get_objects():
#     before[type(i)] += 1 
# =============================================================================

#keep track of day, plant name and TH for every image ID in range 0-1393
#old days list: Days = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3, 31: 3, 32: 3, 33: 3, 34: 3, 35: 3, 36: 3, 37: 3, 38: 3, 39: 3, 40: 3, 41: 3, 42: 3, 43: 3, 44: 3, 45: 3, 46: 3, 47: 3, 48: 3, 49: 3, 50: 3, 51: 3, 52: 3, 53: 3, 54: 3, 55: 3, 56: 3, 57: 3, 58: 3, 59: 3, 60: 3, 61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 3, 72: 3, 73: 3, 74: 3, 75: 3, 76: 3, 77: 3, 78: 3, 79: 3, 80: 3, 81: 3, 82: 3, 83: 3, 84: 3, 85: 3, 86: 3, 87: 3, 88: 3, 89: 3, 90: 3, 91: 3, 92: 3, 93: 3, 94: 3, 95: 3, 96: 4, 97: 4, 98: 4, 99: 4, 100: 4, 101: 4, 102: 4, 103: 4, 104: 4, 105: 4, 106: 4, 107: 4, 108: 4, 109: 4, 110: 4, 111: 4, 112: 4, 113: 4, 114: 4, 115: 4, 116: 4, 117: 4, 118: 4, 119: 4, 120: 4, 121: 4, 122: 4, 123: 4, 124: 4, 125: 4, 126: 4, 127: 4, 128: 4, 129: 4, 130: 4, 131: 4, 132: 4, 133: 4, 134: 4, 135: 4, 136: 4, 137: 4, 138: 4, 139: 4, 140: 4, 141: 4, 142: 4, 143: 4, 144: 4, 145: 4, 146: 4, 147: 4, 148: 4, 149: 4, 150: 4, 151: 4, 152: 4, 153: 4, 154: 4, 155: 4, 156: 4, 157: 4, 158: 4, 159: 4, 160: 4, 161: 4, 162: 4, 163: 4, 164: 4, 165: 4, 166: 4, 167: 4, 168: 4, 169: 4, 170: 4, 171: 4, 172: 4, 173: 4, 174: 4, 175: 4, 176: 4, 177: 4, 178: 4, 179: 4, 180: 4, 181: 4, 182: 4, 183: 4, 184: 4, 185: 4, 186: 4, 187: 4, 188: 4, 189: 4, 190: 4, 191: 4, 192: 5, 193: 5, 194: 5, 195: 5, 196: 5, 197: 5, 198: 5, 199: 5, 200: 5, 201: 5, 202: 5, 203: 5, 204: 5, 205: 5, 206: 5, 207: 5, 208: 5, 209: 5, 210: 5, 211: 5, 212: 5, 213: 5, 214: 5, 215: 5, 216: 5, 217: 5, 218: 5, 219: 5, 220: 5, 221: 5, 222: 5, 223: 5, 224: 5, 225: 5, 226: 5, 227: 5, 228: 5, 229: 5, 230: 5, 231: 5, 232: 5, 233: 5, 234: 5, 235: 5, 236: 5, 237: 5, 238: 5, 239: 5, 240: 5, 241: 5, 242: 5, 243: 5, 244: 5, 245: 5, 246: 5, 247: 5, 248: 5, 249: 5, 250: 5, 251: 5, 252: 5, 253: 5, 254: 5, 255: 5, 256: 5, 257: 5, 258: 5, 259: 5, 260: 5, 261: 5, 262: 5, 263: 5, 264: 5, 265: 5, 266: 5, 267: 5, 268: 5, 269: 5, 270: 5, 271: 5, 272: 5, 273: 5, 274: 5, 275: 5, 276: 5, 277: 5, 278: 5, 279: 5, 280: 5, 281: 5, 282: 5, 283: 5, 284: 5, 285: 5, 286: 5, 287: 5, 288: 6, 289: 6, 290: 6, 291: 6, 292: 6, 293: 6, 294: 6, 295: 6, 296: 6, 297: 6, 298: 6, 299: 6, 300: 6, 301: 6, 302: 6, 303: 6, 304: 6, 305: 6, 306: 6, 307: 6, 308: 6, 309: 6, 310: 6, 311: 6, 312: 6, 313: 6, 314: 6, 315: 6, 316: 6, 317: 6, 318: 6, 319: 6, 320: 6, 321: 6, 322: 6, 323: 6, 324: 6, 325: 6, 326: 6, 327: 6, 328: 6, 329: 6, 330: 6, 331: 6, 332: 6, 333: 6, 334: 6, 335: 6, 336: 6, 337: 6, 338: 6, 339: 6, 340: 6, 341: 6, 342: 6, 343: 6, 344: 6, 345: 6, 346: 6, 347: 6, 348: 6, 349: 6, 350: 6, 351: 6, 352: 6, 353: 6, 354: 6, 355: 6, 356: 6, 357: 6, 358: 6, 359: 6, 360: 6, 361: 6, 362: 6, 363: 6, 364: 6, 365: 6, 366: 6, 367: 6, 368: 6, 369: 6, 370: 6, 371: 6, 372: 6, 373: 6, 374: 6, 375: 6, 376: 6, 377: 6, 378: 6, 379: 6, 380: 6, 381: 6, 382: 6, 383: 6, 384: 7, 385: 7, 386: 7, 387: 7, 388: 7, 389: 7, 390: 7, 391: 7, 392: 7, 393: 7, 394: 7, 395: 7, 396: 7, 397: 7, 398: 7, 399: 7, 400: 7, 401: 7, 402: 7, 403: 7, 404: 7, 405: 7, 406: 7, 407: 7, 408: 7, 409: 7, 410: 7, 411: 7, 412: 7, 413: 7, 414: 7, 415: 7, 416: 7, 417: 7, 418: 7, 419: 7, 420: 7, 421: 7, 422: 7, 423: 7, 424: 7, 425: 7, 426: 7, 427: 7, 428: 7, 429: 7, 430: 7, 431: 7, 432: 7, 433: 7, 434: 7, 435: 7, 436: 7, 437: 7, 438: 7, 439: 7, 440: 7, 441: 7, 442: 7, 443: 7, 444: 7, 445: 7, 446: 7, 447: 7, 448: 7, 449: 7, 450: 7, 451: 7, 452: 7, 453: 7, 454: 7, 455: 7, 456: 7, 457: 7, 458: 7, 459: 7, 460: 7, 461: 7, 462: 7, 463: 7, 464: 7, 465: 7, 466: 7, 467: 7, 468: 7, 469: 7, 470: 7, 471: 7, 472: 7, 473: 7, 474: 7, 475: 7, 476: 7, 477: 7, 478: 7, 479: 7, 480: 8, 481: 8, 482: 8, 483: 8, 484: 8, 485: 8, 486: 8, 487: 8, 488: 8, 489: 8, 490: 8, 491: 8, 492: 8, 493: 8, 494: 8, 495: 8, 496: 8, 497: 8, 498: 8, 499: 8, 500: 8, 501: 8, 502: 8, 503: 8, 504: 8, 505: 8, 506: 8, 507: 8, 508: 8, 509: 8, 510: 8, 511: 8, 512: 8, 513: 8, 514: 8, 515: 8, 516: 8, 517: 8, 518: 8, 519: 8, 520: 8, 521: 8, 522: 8, 523: 8, 524: 8, 525: 8, 526: 8, 527: 8, 528: 8, 529: 8, 530: 8, 531: 8, 532: 8, 533: 8, 534: 8, 535: 8, 536: 8, 537: 8, 538: 8, 539: 8, 540: 8, 541: 8, 542: 8, 543: 8, 544: 8, 545: 8, 546: 8, 547: 8, 548: 8, 549: 8, 550: 8, 551: 8, 552: 8, 553: 8, 554: 8, 555: 8, 556: 8, 557: 8, 558: 8, 559: 8, 560: 8, 561: 8, 562: 8, 563: 8, 564: 8, 565: 8, 566: 8, 567: 8, 568: 8, 569: 8, 570: 8, 571: 8, 572: 8, 573: 8, 574: 8, 575: 9, 576: 9, 577: 9, 578: 9, 579: 9, 580: 9, 581: 9, 582: 9, 583: 9, 584: 9, 585: 9, 586: 9, 587: 9, 588: 9, 589: 9, 590: 9, 591: 9, 592: 9, 593: 9, 594: 9, 595: 9, 596: 9, 597: 9, 598: 9, 599: 9, 600: 9, 601: 9, 602: 9, 603: 9, 604: 9, 605: 9, 606: 9, 607: 9, 608: 9, 609: 9, 610: 9, 611: 9, 612: 9, 613: 9, 614: 9, 615: 9, 616: 9, 617: 9, 618: 9, 619: 9, 620: 9, 621: 9, 622: 9, 623: 9, 624: 9, 625: 9, 626: 9, 627: 9, 628: 9, 629: 9, 630: 9, 631: 9, 632: 9, 633: 9, 634: 9, 635: 9, 636: 9, 637: 9, 638: 9, 639: 9, 640: 9, 641: 9, 642: 9, 643: 9, 644: 9, 645: 9, 646: 9, 647: 9, 648: 9, 649: 9, 650: 9, 651: 9, 652: 9, 653: 9, 654: 9, 655: 9, 656: 9, 657: 9, 658: 9, 659: 9, 660: 9, 661: 9, 662: 9, 663: 9, 664: 9, 665: 9, 666: 9, 667: 9, 668: 9, 669: 9, 670: 9, 671: 10, 672: 10, 673: 10, 674: 10, 675: 10, 676: 10, 677: 10, 678: 10, 679: 10, 680: 10, 681: 10, 682: 10, 683: 10, 684: 10, 685: 10, 686: 10, 687: 10, 688: 10, 689: 10, 690: 10, 691: 10, 692: 10, 693: 10, 694: 10, 695: 10, 696: 10, 697: 10, 698: 10, 699: 10, 700: 10, 701: 10, 702: 10, 703: 10, 704: 10, 705: 10, 706: 10, 707: 10, 708: 10, 709: 10, 710: 10, 711: 10, 712: 10, 713: 10, 714: 10, 715: 10, 716: 10, 717: 10, 718: 10, 719: 10, 720: 10, 721: 10, 722: 10, 723: 10, 724: 10, 725: 10, 726: 10, 727: 10, 728: 10, 729: 10, 730: 10, 731: 10, 732: 10, 733: 10, 734: 10, 735: 10, 736: 10, 737: 10, 738: 10, 739: 10, 740: 10, 741: 10, 742: 10, 743: 10, 744: 10, 745: 10, 746: 10, 747: 10, 748: 10, 749: 10, 750: 10, 751: 10, 752: 10, 753: 10, 754: 10, 755: 10, 756: 10, 757: 10, 758: 10, 759: 10, 760: 10, 761: 10, 762: 10, 763: 11, 764: 11, 765: 11, 766: 11, 767: 11, 768: 11, 769: 11, 770: 11, 771: 11, 772: 11, 773: 11, 774: 11, 775: 11, 776: 11, 777: 11, 778: 11, 779: 11, 780: 11, 781: 11, 782: 11, 783: 11, 784: 11, 785: 11, 786: 11, 787: 11, 788: 11, 789: 11, 790: 11, 791: 11, 792: 11, 793: 11, 794: 11, 795: 11, 796: 11, 797: 11, 798: 11, 799: 11, 800: 11, 801: 11, 802: 11, 803: 11, 804: 11, 805: 11, 806: 11, 807: 11, 808: 11, 809: 11, 810: 11, 811: 11, 812: 11, 813: 11, 814: 11, 815: 11, 816: 11, 817: 11, 818: 11, 819: 11, 820: 11, 821: 11, 822: 11, 823: 11, 824: 11, 825: 11, 826: 11, 827: 11, 828: 11, 829: 11, 830: 11, 831: 11, 832: 11, 833: 11, 834: 11, 835: 11, 836: 11, 837: 11, 838: 11, 839: 11, 840: 11, 841: 11, 842: 11, 843: 11, 844: 11, 845: 11, 846: 11, 847: 11, 848: 11, 849: 12, 850: 12, 851: 12, 852: 12, 853: 12, 854: 12, 855: 12, 856: 12, 857: 12, 858: 12, 859: 12, 860: 12, 861: 12, 862: 12, 863: 12, 864: 12, 865: 12, 866: 12, 867: 12, 868: 12, 869: 12, 870: 12, 871: 12, 872: 12, 873: 12, 874: 13, 875: 12, 876: 12, 877: 13, 878: 12, 879: 12, 880: 12, 881: 12, 882: 12, 883: 12, 884: 13, 885: 12, 886: 12, 887: 12, 888: 12, 889: 12, 890: 12, 891: 12, 892: 13, 893: 12, 894: 12, 895: 13, 896: 12, 897: 12, 898: 12, 899: 13, 900: 12, 901: 12, 902: 12, 903: 12, 904: 12, 905: 13, 906: 12, 907: 12, 908: 12, 909: 12, 910: 12, 911: 12, 912: 12, 913: 13, 914: 12, 915: 12, 916: 12, 917: 12, 918: 12, 919: 12, 920: 13, 921: 12, 922: 12, 923: 12, 924: 12, 925: 13, 926: 13, 927: 12, 928: 12, 929: 13, 930: 12, 931: 13, 932: 12, 933: 13, 934: 12, 935: 13, 936: 12, 937: 12, 938: 12, 939: 13, 940: 12, 941: 12, 942: 13, 943: 12, 944: 12, 945: 13, 946: 12, 947: 12, 948: 12, 949: 13, 950: 14, 951: 14, 952: 14, 953: 14, 954: 14, 955: 14, 956: 14, 957: 14, 958: 14, 959: 14, 960: 14, 961: 14, 962: 14, 963: 14, 964: 14, 965: 14, 966: 14, 967: 14, 968: 14, 969: 14, 970: 13, 971: 15, 972: 13, 973: 13, 974: 15, 975: 13, 976: 13, 977: 15, 978: 13, 979: 13, 980: 15, 981: 15, 982: 13, 983: 13, 984: 13, 985: 13, 986: 13, 987: 15, 988: 13, 989: 13, 990: 15, 991: 13, 992: 15, 993: 15, 994: 13, 995: 13, 996: 15, 997: 13, 998: 13, 999: 13, 1000: 13, 1001: 13, 1002: 13, 1003: 13, 1004: 13, 1005: 13, 1006: 13, 1007: 13, 1008: 13, 1009: 15, 1010: 13, 1011: 13, 1012: 13, 1013: 13, 1014: 15, 1015: 15, 1016: 13, 1017: 13, 1018: 15, 1019: 13, 1020: 15, 1021: 13, 1022: 15, 1023: 13, 1024: 15, 1025: 13, 1026: 13, 1027: 13, 1028: 13, 1029: 15, 1030: 13, 1031: 15, 1032: 13, 1033: 13, 1034: 15, 1035: 13, 1036: 13, 1037: 15, 1038: 16, 1039: 16, 1040: 16, 1041: 16, 1042: 16, 1043: 16, 1044: 16, 1045: 16, 1046: 16, 1047: 16, 1048: 16, 1049: 16, 1050: 16, 1051: 16, 1052: 17, 1053: 17, 1054: 14, 1055: 14, 1056: 17, 1057: 14, 1058: 14, 1059: 14, 1060: 14, 1061: 17, 1062: 14, 1063: 14, 1064: 14, 1065: 14, 1066: 14, 1067: 14, 1068: 14, 1069: 14, 1070: 14, 1071: 14, 1072: 17, 1073: 14, 1074: 14, 1075: 14, 1076: 14, 1077: 14, 1078: 14, 1079: 14, 1080: 14, 1081: 14, 1082: 14, 1083: 14, 1084: 14, 1085: 14, 1086: 14, 1087: 14, 1088: 14, 1089: 14, 1090: 14, 1091: 14, 1092: 14, 1093: 14, 1094: 14, 1095: 14, 1096: 14, 1097: 14, 1098: 17, 1099: 14, 1100: 14, 1101: 14, 1102: 14, 1103: 14, 1104: 14, 1105: 14, 1106: 14, 1107: 17, 1108: 14, 1109: 14, 1110: 14, 1111: 14, 1112: 18, 1113: 18, 1114: 18, 1115: 18, 1116: 19, 1117: 19, 1118: 19, 1119: 19, 1120: 19, 1121: 19, 1122: 19, 1123: 19, 1124: 19, 1125: 20, 1126: 15, 1127: 15, 1128: 20, 1129: 15, 1130: 15, 1131: 15, 1132: 20, 1133: 15, 1134: 20, 1135: 20, 1136: 15, 1137: 15, 1138: 15, 1139: 15, 1140: 15, 1141: 15, 1142: 20, 1143: 15, 1144: 15, 1145: 20, 1146: 15, 1147: 15, 1148: 15, 1149: 20, 1150: 15, 1151: 20, 1152: 15, 1153: 15, 1154: 15, 1155: 15, 1156: 15, 1157: 20, 1158: 15, 1159: 15, 1160: 15, 1161: 15, 1162: 15, 1163: 15, 1164: 15, 1165: 15, 1166: 15, 1167: 15, 1168: 15, 1169: 15, 1170: 20, 1171: 15, 1172: 20, 1173: 15, 1174: 15, 1175: 15, 1176: 15, 1177: 15, 1178: 15, 1179: 20, 1180: 15, 1181: 15, 1182: 15, 1183: 15, 1184: 20, 1185: 21, 1186: 16, 1187: 16, 1188: 16, 1189: 21, 1190: 16, 1191: 21, 1192: 21, 1193: 16, 1194: 16, 1195: 16, 1196: 16, 1197: 16, 1198: 21, 1199: 16, 1200: 16, 1201: 16, 1202: 16, 1203: 16, 1204: 16, 1205: 16, 1206: 21, 1207: 16, 1208: 16, 1209: 16, 1210: 16, 1211: 16, 1212: 16, 1213: 16, 1214: 16, 1215: 16, 1216: 16, 1217: 16, 1218: 16, 1219: 16, 1220: 16, 1221: 16, 1222: 16, 1223: 16, 1224: 16, 1225: 16, 1226: 16, 1227: 21, 1228: 16, 1229: 16, 1230: 21, 1231: 16, 1232: 16, 1233: 16, 1234: 16, 1235: 16, 1236: 16, 1237: 16, 1238: 21, 1239: 16, 1240: 16, 1241: 16, 1242: 16, 1243: 21, 1244: 17, 1245: 17, 1246: 17, 1247: 17, 1248: 17, 1249: 17, 1250: 17, 1251: 17, 1252: 17, 1253: 17, 1254: 17, 1255: 17, 1256: 22, 1257: 17, 1258: 17, 1259: 17, 1260: 17, 1261: 17, 1262: 17, 1263: 17, 1264: 17, 1265: 17, 1266: 17, 1267: 17, 1268: 17, 1269: 17, 1270: 17, 1271: 17, 1272: 22, 1273: 17, 1274: 17, 1275: 17, 1276: 17, 1277: 17, 1278: 17, 1279: 17, 1280: 17, 1281: 17, 1282: 22, 1283: 17, 1284: 22, 1285: 22, 1286: 17, 1287: 17, 1288: 17, 1289: 17, 1290: 17, 1291: 17, 1292: 17, 1293: 22, 1294: 17, 1295: 17, 1296: 17, 1297: 17, 1298: 2, 1299: 2, 1300: 2, 1301: 2, 1302: 2, 1303: 2, 1304: 2, 1305: 2, 1306: 2, 1307: 2, 1308: 2, 1309: 2, 1310: 2, 1311: 2, 1312: 2, 1313: 2, 1314: 2, 1315: 2, 1316: 2, 1317: 2, 1318: 2, 1319: 2, 1320: 2, 1321: 2, 1322: 2, 1323: 2, 1324: 2, 1325: 2, 1326: 2, 1327: 2, 1328: 2, 1329: 2, 1330: 2, 1331: 2, 1332: 2, 1333: 2, 1334: 2, 1335: 2, 1336: 2, 1337: 2, 1338: 2, 1339: 2, 1340: 2, 1341: 2, 1342: 2, 1343: 2, 1344: 2, 1345: 2, 1346: 2, 1347: 2, 1348: 2, 1349: 2, 1350: 2, 1351: 2, 1352: 2, 1353: 2, 1354: 2, 1355: 2, 1356: 2, 1357: 2, 1358: 2, 1359: 2, 1360: 2, 1361: 2, 1362: 2, 1363: 2, 1364: 2, 1365: 2, 1366: 2, 1367: 2, 1368: 2, 1369: 2, 1370: 2, 1371: 2, 1372: 2, 1373: 2, 1374: 2, 1375: 2, 1376: 2, 1377: 2, 1378: 2, 1379: 2, 1380: 2, 1381: 2, 1382: 2, 1383: 2, 1384: 2, 1385: 2, 1386: 2, 1387: 2, 1388: 2, 1389: 2, 1390: 2, 1391: 2, 1392: 2}
Days = {817:2, 1153:3, 1489:4, 1825:5, 2161:6 ,2497:7 ,2833:8, 3169:9, 3505:10, 3841:11, 4261:13, 4681:15, 5101:17, 5605:20, 5941:21, 6277:22, 821:2, 1157:3, 1493:4, 1829:5, 2165:6, 2501:7, 2837:8, 3173:9, 3509:10, 3845:11, 4068:12, 4265:13, 4488:14, 4908:16, 5609:20, 6281:22, 816:2,1152:3, 1488:4, 1824:5, 2160:6, 2496:7, 2832:8, 3168:9, 3504:10, 3840:11, 4260:13, 4680:15, 5100:17, 5604:20, 5940:21, 6276:22, 818:2, 1154:3, 1490:4, 1826:5, 2162:6, 2498:7, 2834:8, 3170:9, 3506:10, 3842:11, 4262:13, 4682:15, 5102:17, 5606:20, 5942:21, 6278:22, 779:2, 1115:3, 1451:4, 1787:5, 2123:6, 2459:7, 2795:8, 3131:9, 3467:10, 3803:11, 4223:13, 4643:15, 5063:17, 5567:20, 5903:21, 6239:22, 803:2, 1139:3, 1475:4, 1811:5, 2147:6, 2483:7, 2819:8, 3155:9, 3491:10, 3827:11, 4247:13, 4667:15, 5087:17, 5591:20, 5927:21, 6263:22} 

Names = {817:'T89P103', 1153:'T89P103', 1489:'T89P103', 1825:'T89P103', 2161:'T89P103', 2497:'T89P103', 2833:'T89P103', 3169:'T89P103', 3505:'T89P103', 3841:'T89P103', 4261:'T89P103', 4681:'T89P103', 5101:'T89P103', 5605:'T89P103', 5941:'T89P103', 6277:'T89P103', 821:'T89P133', 1157:'T89P133', 1493:'T89P133', 1829:'T89P133', 2165:'T89P133', 2501:'T89P133', 2837:'T89P133', 3173:'T89P133', 3509:'T89P133', 3845:'T89P133', 4068:'T89P133', 4265:'T89P133', 4488:'T89P133', 4908:'T89P133', 5609:'T89P133', 6281:'T89P133', 816:'T89P100',1152:'T89P100', 1488:'T89P100', 1824:'T89P100', 2160:'T89P100', 2496:'T89P100', 2832:'T89P100', 3168:'T89P100', 3504:'T89P100', 3840:'T89P100', 4260:'T89P100', 4680:'T89P100', 5100:'T89P100', 5604:'T89P100', 5940:'T89P100', 6276:'T89P100', 818:'T89P108', 1154:'T89P108', 1490:'T89P108', 1826:'T89P108', 2162:'T89P108', 2498:'T89P108', 2834:'T89P108', 3170:'T89P108', 3506:'T89P108', 3842:'T89P108', 4262:'T89P108', 4682:'T89P108', 5102:'T89P108', 5606:'T89P108', 5942:'T89P108', 6278:'T89P108', 779:' HpZTL11L5P244', 1115:' HpZTL11L5P244', 1451:' HpZTL11L5P244', 1787:' HpZTL11L5P244', 2123:' HpZTL11L5P244', 2459:' HpZTL11L5P244', 2795:' HpZTL11L5P244', 3131:' HpZTL11L5P244', 3467:' HpZTL11L5P244', 3803:' HpZTL11L5P244', 4223:' HpZTL11L5P244', 4643:' HpZTL11L5P244', 5063:' HpZTL11L5P244', 5567:' HpZTL11L5P244', 5903:' HpZTL11L5P244', 6239:' HpZTL11L5P244', 803: ' HpZTL11L7P260', 1139 :' HpZTL11L7P260', 1475:' HpZTL11L7P260', 1811:' HpZTL11L7P260', 2147:' HpZTL11L7P260', 2483:' HpZTL11L7P260', 2819:' HpZTL11L7P260', 3155:' HpZTL11L7P260', 3491:' HpZTL11L7P260', 3827:' HpZTL11L7P260', 4247:' HpZTL11L7P260', 4667:' HpZTL11L7P260', 5087:' HpZTL11L7P260', 5591:' HpZTL11L7P260', 5927:' HpZTL11L7P260', 6263:' HpZTL11L7P260'} 

THs = {817:1.9, 1153: 1.9, 1489: 1.9, 1825: 1.9, 2161: 1.9 ,2497: 1.9 ,2833: 1.9, 3169: 1.9, 3505: 1.9, 3841: 1.9, 4261: 1.9, 4681: 1.9, 5101: 1.9, 5605: 1.9, 5941: 1.9, 6277: 1.9, 821:0.3, 1157: 0.3, 1493: 0.3, 1829: 0.3, 2165: 0.3, 2501: 0.3, 2837: 0.3, 3173: 0.3, 3509:0.3, 3845: 0.3, 4068: 0.3, 4265: 0.3, 4488: 0.3, 4908: 0.3, 5609: 0.3, 6281: 0.3, 816:0.7,1152:0.7, 1488:0.7, 1824:0.7, 2160:0.7, 2496:0.7, 2832:0.7, 3168:0.7, 3504:0.7, 3840:0.7, 4260:0.7, 4680:0.7, 5100:0.7, 5604:0.7, 5940:0.7, 6276:0.7, 818:0.6, 1154:0.6, 1490:0.6, 1826:0.6, 2162:0.6, 2498:0.6, 2834:0.6, 3170:0.6, 3506:0.6, 3842:0.6, 4262:0.6, 4682:0.6, 5102:0.6, 5606:0.6, 5942:0.6, 6278:0.6, 779:1.9, 1115:1.9, 1451:1.9, 1787:1.9, 2123:1.9, 2459:1.9, 2795:1.9, 3131:1.9, 3467:1.9, 3803:1.9, 4223:1.9, 4643:1.9, 5063:1.9, 5567:1.9, 5903:1.9, 6239:1.9, 803:1.9, 1139:1.9, 1475:1.9, 1811:1.9, 2147:1.9, 2483:1.9, 2819:1.9, 3155:1.9, 3491:1.9, 3827:1.9, 4247:1.9, 4667:1.9, 5087:1.9, 5591:1.9, 5927:1.9, 6263:1.9} 

class CocoLikeDataset(utils.Dataset):
    def load_data(self, annotation_json, images_dir):
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        print("-", len(coco_json['annotations']),"leaves in this dataset")
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        print("-", image_id+1, "images added")
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids

print("done importing packages")

print("check if GPU available:")
import tensorflow as tf
tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

#---------------------DIRECTORIES--------------------------------------------------

# Root directory of the project
ROOT_DIR = "C:\\Users\\Exjobb\\Downloads\\Code_folder"
DIR = "F:\\from_Disk"
MODEL_DIR = os.path.join(DIR, "logs")
WEIGHTS_PATH= os.path.join(MODEL_DIR , "default", "mask_rcnn_object_0100.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(DIR, "preprocessed_data", "RGB")
THERMAL_DIR = os.path.join(DIR, "preprocessed_data", "csv")
THERMAL_IMG_DIR = os.path.join(DIR, "preprocessed_data", "thermal")
RESULTS_DIR =  os.path.join(DIR, "temp_results")
fc.delete_and_create_directory(RESULTS_DIR)
HIST_DIR = os.path.join(RESULTS_DIR, "hists")
fc.delete_and_create_directory(HIST_DIR)
MASK_DIR = os.path.join(RESULTS_DIR,"masks")
fc.delete_and_create_directory(MASK_DIR)
CENTER_DIR = os.path.join(RESULTS_DIR,"centers")
fc.delete_and_create_directory(CENTER_DIR)
MASK_THERMAL_DIR = os.path.join(RESULTS_DIR,"mask_thermal")
fc.delete_and_create_directory(MASK_THERMAL_DIR)
BG_DIR = os.path.join(RESULTS_DIR,"BGs")
fc.delete_and_create_directory(BG_DIR)
#DATA_PATH = os.path.join("F:\\Hannah", "dataset")






#----------------------CONFIGURATIONS-------------------------------------------

class_names = ['BG', 'leaf']
#colors = random_colors(len(class_names))

mask_color = (255, 255, 255)

class CustomConfig(Config):

    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    DETECTION_MIN_CONFIDENCE = 0.7
    STEPS_PER_EPOCH = 10
    



#---------------MODEL LOADING-------------------


print("loading model...")

config = CustomConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
print("done!")
print("Loading weights ", WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH, by_name=True)

#class_names = ['BG', 'leaves']



#-----------------PROCESSING FUNCTIONS----------------------------------

    
def color_splash(img, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(img)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        #mask = (np.sum(mask, -1, keepdims=True) >= 1)
        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
        splash = np.where(mask, img, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash
    
    
def detecting(model, image, name):
    """detect instances, classify, draw boxes and mask
    If no instances are detected gives empty outputs
    Input: model, image to detect and its name
    Output: class IDs, bounding boxes, masks as lists converted to use it more easily but also r the usual detection output
    """
    try:
      class_ids, boxes, masks, r= detect_contours_maskrcnn_with_original(model, image) #masks are given as coordinates (x,y)
    except:
      print("No leaves could be detected, plant died probably")
      class_ids, boxes, masks,r = [],[],[], []
    print("--", len(masks), "leaves detected in image", int(re.findall(r'\d+', name)[0]))
 
    return class_ids, boxes, masks , r 

def create_final_image_with_leaf_centers(class_ids, boxes, masks, final_img, mask_img):
    """
    Create image of all detected leaves, store the leaf centers

    Parameters
    ----------
    class_ids, boxes, masks : generated from detecting
    final_img : empty copy of original image to store all detected masks
    mask_img : empty copy of original image used for leaf center calculation

    Returns
    -------
    final_img: image of all leaf masks, 
    leaf_centers: list
        STORES COORDINATES OF THE LEAVES

    """ 
    
    leaf_centers = []
    
    for class_id, box, object_contours in zip(class_ids, boxes, masks):
        mask_img[:]= 0
        final_img = draw_mask(final_img, [object_contours], mask_color)      
        masked_img= draw_mask(mask_img, [object_contours], mask_color[0])
        #calculate center of leaf and store x and y values in leaf_centers
        leaf_centers.append([fc.image_center(masked_img)])
    
    #del mask_img

    return final_img, leaf_centers
    
def get_closest_6_leaves(final_img, leaf_centers):
    """
    Calculate the center of final_img, calculate and store in distances the distance of every
    leaf center that center. Sort descending and keep the indices of the closest 6 leaves

    Parameters
    ----------
    final_img, leaf_centers : generated from create_final_image_with_leaf_centers

    Returns
    -------
    x,y: values
        CENTER OF final_img
    closest6 : list
        INDICES OF THE 6 LEAVES CLOSEST TO THE CENTER OF final_img.

    """
    
    
    x,y = fc.image_center(final_img)
    distances = []
    for coordinates in leaf_centers:
      distances.append((coordinates[0][0]-x)**2 + (coordinates[0][1]-y)**2)#actually squared distance but does not matter
      
    #sort the list descending, keep the indices
    #https://stackoverflow.com/questions/66679020/python-quickest-way-to-sort-list-and-keep-indexes
    sorted_indices = np.array(distances).argsort()
    #print("sorted indices are:", sorted_indices)
    closest6 = sorted_indices[:6]
    if len(sorted_indices) >5:
      assert len(closest6) == 6, print("this is more or less than 6")
    else:
      print("less than 6 images detected")
     
    del final_img
    
    return x,y, closest6

                    
def temperature_analysis(name, class_ids, boxes, masks,final_img, leaf_centers, x,y, closest6, mask6, thermal_data, R):
    """
    For the 6 closest leaves
    Adjust mask and thermal data shape, overlap both and analyze temperature with fc.analyze_thermal_values.
    Print the leaf max, min, average T and size and save the histogram at HIST_DIR, name,"hist{}.png".
    Calculate the average of the 6 leaves for every value and display final_img with the image center 
    and the 6 leaf centers.

    Parameters
    ----------
    name: file name to safe the overlapped thermal data and mask. Including .png or .jpg
    class_ids, boxes, masks : GENERATED FROM detecting.
    final_img, leaf_centers : GENERATED FROM create_final_image_with_leaf_centers.
    closest6 : GENERATED FROM get_closest_6_leaves.
    img : cv2.imread image, ORIGINAL IMAGE USED FOR DETECTION.
    thermal_data: pcv.readimage, THERMAL DATA TO img
    r: original mask, not preprocessed

    Returns
    -------
    av_avtemp,av_maxtemp,av_mintemp,av_size: VALUES
        AVERAGE VALUES OF MEAN, MAX, MIN T AND SIZE OF THE 6 LEAVES

    """
    
    # max_temps, min_temps, average_temps, sizes = [],[],[],[]

# =============================================================================
#     #all masks
#     final = final_img[:,:,0]
#     final = cv2.resize(final, (thermal_data.shape[1], thermal_data.shape[0]))
# =============================================================================
    
    #colorsplash mask with thermal rgb image
    thermal_colored = cv2.imread(os.path.join(THERMAL_IMG_DIR, name ))
    thermal_colored = cv2.cvtColor(thermal_colored, cv2.COLOR_RGB2BGR)
    thermal_resized = cv2.resize(thermal_colored, (final_img.shape[1], final_img.shape[0]))
    splash = visualize.color_splash(thermal_resized, R['masks'])
    file_name = os.path.join(MASK_THERMAL_DIR, "mask_ther{}".format(name ))
    skimage.io.imsave(file_name, splash)
# =============================================================================
#     final_resized = cv2.resize(final, (thermal_colored.shape[1], thermal_colored.shape[0]))
#     splashed = color_splash(thermal_colored, final_resized)
#     cv2.imwrite(os.path.join(RESULTS_DIR, "mask_ther{}".format(name )), splashed)
# =============================================================================
        
    plt.figure(frameon=False)#BG mask plot
    
    #analyze the whole plant
    mask_img = cv2.resize(final_img, (thermal_data.shape[1], thermal_data.shape[0])) 
    mask_img = mask_img[:,:,0]
    #assert mask_img.shape == thermal_data.shape, "shapes are not equal, resize or crop!"        
    results = fc.analyze_thermal_values(thermal_data, mask_img, histplot= False, label="default");
    
    # Access data stored out from analyze_thermal_values
    whole_average_temp = pcv.outputs.observations['default']['mean_temp']['value']
    background_mask = 128-mask_img
    plt.imshow(background_mask)
    plt.axis('off')
    plt.savefig(os.path.join(BG_DIR, "background{}".format(name)), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    plt.figure(frameon=False) #plot of mask with centers of the 6 closest
    
    results = fc.analyze_thermal_values(thermal_data, background_mask, histplot= False, label="default");
    background_temp = pcv.outputs.observations['default']['mean_temp']['value']

    

    #r = [[],[],[]]
    for j,count in zip(closest6, range(6)):
        object_contours = masks[j]

        
        mask6= draw_mask(mask6, [object_contours], mask_color)

        #single leaf mask to overlap with thermal data and analyze leaf T
        mask_img = final_img.copy()
        mask_img[:]= 0
        mask_img= draw_mask(mask_img, [object_contours], mask_color[0])
        mask_img = mask_img[:,:,0]
        mask_img = cv2.resize(mask_img, (thermal_data.shape[1], thermal_data.shape[0])) 
        #assert mask_img.shape == thermal_data.shape, "shapes are not equal, resize or crop!"        
        results = fc.analyze_thermal_values(thermal_data, mask_img, histplot= False, label="default");
        
        # Access data stored out from analyze_thermal_values
        average_temp = pcv.outputs.observations['default']['mean_temp']['value']
        max_temp = pcv.outputs.observations['default']['max_temp']['value']
        min_temp = pcv.outputs.observations['default']['min_temp']['value']
        size = cv2.countNonZero(mask_img)
        leaf_id = count
        nr = int(re.findall(r'\d+', name)[0])
        th = THs[nr]
        day = Days[nr]
        genotype = Names[nr]
        if "T89" in genotype:
            genotype = "T89"
        elif "11L5" in genotype:
            genotype = "11L5"
        elif "11L7" in genotype:
            genotype = "11L7"
        
        
        filewriter.writerow([nr, genotype, th, day, leaf_id, average_temp,max_temp, min_temp, size,whole_average_temp,background_temp])
        
# =============================================================================
#         max_temps.append(max_temp)
#         min_temps.append(min_temp)
#         average_temps.append(average_temp)
#         sizes.append(size)
# =============================================================================
        
        print("Leaf of size {3} has average temperature: {0:.2f}, min temperature: {1:.2f}, max temperature: {2:.2f}".format(average_temp, min_temp,max_temp, size))
        #save hist ggplot for every leaf
        results.save(os.path.join(HIST_DIR, name,"hist{}.png".format(j)), dpi=600, verbose = False)
        
        #mask with center marks to see which leaves were taken    
        plt.plot(leaf_centers[j][0][0],leaf_centers[j][0][1] , 'r+', linewidth=2, markersize=12)
        del mask_img
    plt.plot(x,y , 'r+', linewidth=2, markersize=12)
    plt.imshow(final_img)
    plt.axis('off')
    plt.savefig(os.path.join(CENTER_DIR, "all_leaves{}".format(name)), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    
    mask6 = mask6[:,:,0]
    cv2.imwrite(os.path.join(MASK_DIR, "6mask{}".format(name) ),mask6 )
    
# =============================================================================
#     av_avtemp = round(sum(average_temps) / len(average_temps),2)
#     av_maxtemp = round(sum(max_temps) / len(max_temps),2)
#     av_mintemp = round(sum(min_temps) / len(min_temps),2)
#     av_size = round(sum(sizes) / len(sizes),2)
# =============================================================================
    

    del final_img

    return average_temp,max_temp, min_temp, size

def run_all(name, name_csv):
    nr = int(re.findall(r'\d+', name)[0])
    img = cv2.imread(os.path.join(IMAGE_DIR, name))
    final_img = img.copy() 
    final_img[:]= 0
    mask_img = img.copy()
    mask_img[:]= 0
    #the 6 closes leaves mask
    mask6 = img.copy()
    mask6[:]= 0
    

    thermal_data,path,filename = pcv.readimage(filename=os.path.join(THERMAL_DIR, name_csv), mode="csv")
    class_ids, boxes, masks,r  = detecting(model, img, name)       
    final_img, leaf_centers = create_final_image_with_leaf_centers(class_ids, boxes, masks, final_img, mask_img)

     
    try:
        x,y, closest6 = get_closest_6_leaves(final_img, leaf_centers)
        fc.delete_and_create_directory(os.path.join(HIST_DIR, name))
        av_avtemp,av_maxtemp,av_mintemp,av_size = temperature_analysis(name,class_ids, boxes, masks,final_img, leaf_centers, x,y, closest6, mask6, thermal_data, r)
      
      
    except:
        th = THs[nr]
        day = Days[nr]
        genotype = Names[nr]
        filewriter.writerow([nr,genotype, th, day,"NaN", "NaN", "NaN", "NaN", "NaN","NaN","NaN"])
        # av_avtemp,av_maxtemp,av_mintemp,av_size = "NaN", "NaN", "NaN", "NaN"
     
    #print ("Plants average temperature = {0}, max temp = {1}, min temp = {2}, size = {3}".format(av_avtemp,  av_maxtemp,av_mintemp, av_size ))
    
    
    #cv2.polylines(mask_img, [object_contours], True, mask_color, 1)
    cv2.imwrite(os.path.join(MASK_DIR, "allmask{}".format(name) ),final_img )
    cv2.destroyAllWindows()
    plt.close()
    gc.collect()
    del final_img
#-----------------ACTUAL EVALUATION---------------------------------------

#store the data averaged over the 6 closest leaves for every plant
with open(RESULTS_DIR+'\\final_results.csv', 'w', newline='') as csvfile:
  filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  filewriter.writerow(['id', 'genotype', 'TH', 'day', 'leafid', 'average', 'maxx', 'minn', 'sizess','avT_whole_plant', 'background_T'])

  
  for name, name_csv in tqdm(zip(os.listdir(IMAGE_DIR), os.listdir(THERMAL_DIR)),total =len(os.listdir(THERMAL_DIR))):
  #name = "img0.png"
      run_all(name, name_csv)

    
  #Test: read the last final mask in
  #img_m = cv2.imread(os.path.join(MASK_DIR, name))
  #cv2.imshow('Contour', img_m)
  #cv2.waitKey(0)
  cv2.destroyAllWindows()


  
                     