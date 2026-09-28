/*
 * Standalone Implementation of MCU Fault Classifier
 */

#include "fault_classifier.h"
#include <math.h>

// Feature Scaling Parameters (StandardScaler)

static const float SCALER_MEAN[8] = {
    4.755512f, 0.006445f, 4.755524f, 0.019217f, 1.395066f, 0.143770f, 1.403305f, 0.381731f
};

static const float SCALER_SCALE[8] = {
    0.805471f, 0.004631f, 0.805441f, 0.014469f, 0.501172f, 0.020897f, 0.499224f, 0.055747f
};

static const char* FAULT_NAMES[NUM_CLASSES] = {
    "Healthy",
    "ESR Mild",
    "ESR Severe",
    "Switch Deg",
    "Gate Deg",
    "Sensor Fault",
};

static void evaluate_tree_0(const float features[8], float votes[6]) {
    if (features[5] <= -0.116263f) {
        if (features[2] <= -0.589706f) {
            if (features[1] <= 1.371376f) {
                if (features[7] <= -6.791758f) {
                    if (features[1] <= -1.390372f) {
                        votes[0] += 0.082569f;
                        votes[1] += 0.055046f;
                        votes[2] += 0.055046f;
                        votes[3] += 0.110092f;
                        votes[4] += 0.633028f;
                        votes[5] += 0.064220f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[1] <= 1.257547f) {
                        votes[0] += 0.158856f;
                        votes[1] += 0.146148f;
                        votes[2] += 0.081017f;
                        votes[3] += 0.278793f;
                        votes[4] += 0.160445f;
                        votes[5] += 0.174742f;
                    } else {
                        votes[0] += 0.203180f;
                        votes[1] += 0.201413f;
                        votes[2] += 0.075972f;
                        votes[3] += 0.063604f;
                        votes[4] += 0.229682f;
                        votes[5] += 0.226148f;
                    }
                }
            } else {
                if (features[6] <= -1.422389f) {
                    if (features[1] <= 1.402248f) {
                        votes[1] += 0.315789f;
                        votes[2] += 0.684211f;
                    } else {
                        votes[0] += 0.201389f;
                        votes[1] += 0.190972f;
                        votes[2] += 0.159722f;
                        votes[3] += 0.104167f;
                        votes[4] += 0.152778f;
                        votes[5] += 0.190972f;
                    }
                } else {
                    if (features[7] <= -0.404721f) {
                        votes[0] += 0.132576f;
                        votes[1] += 0.071970f;
                        votes[2] += 0.659091f;
                        votes[4] += 0.068182f;
                        votes[5] += 0.068182f;
                    } else {
                        votes[0] += 0.104478f;
                        votes[1] += 0.253731f;
                        votes[2] += 0.358209f;
                        votes[4] += 0.164179f;
                        votes[5] += 0.119403f;
                    }
                }
            }
        } else {
            if (features[0] <= 0.301726f) {
                if (features[6] <= 0.886777f) {
                    if (features[5] <= -0.189499f) {
                        votes[0] += 0.009887f;
                        votes[1] += 0.014124f;
                        votes[2] += 0.057203f;
                        votes[3] += 0.020480f;
                        votes[4] += 0.883475f;
                        votes[5] += 0.014831f;
                    } else {
                        votes[0] += 0.057789f;
                        votes[1] += 0.077889f;
                        votes[2] += 0.560302f;
                        votes[3] += 0.080402f;
                        votes[4] += 0.158291f;
                        votes[5] += 0.065327f;
                    }
                } else {
                    if (features[0] <= 0.109260f) {
                        votes[0] += 0.255255f;
                        votes[1] += 0.195195f;
                        votes[2] += 0.120120f;
                        votes[3] += 0.114114f;
                        votes[4] += 0.009009f;
                        votes[5] += 0.306306f;
                    } else {
                        votes[0] += 0.056818f;
                        votes[1] += 0.017045f;
                        votes[2] += 0.818182f;
                        votes[4] += 0.028409f;
                        votes[5] += 0.079545f;
                    }
                }
            } else {
                if (features[7] <= 0.015132f) {
                    votes[2] += 1.000000f;
                } else {
                    if (features[6] <= 0.204835f) {
                        votes[0] += 0.001145f;
                        votes[1] += 0.002864f;
                        votes[2] += 0.005155f;
                        votes[4] += 0.989691f;
                        votes[5] += 0.001145f;
                    } else {
                        votes[0] += 0.010417f;
                        votes[1] += 0.006629f;
                        votes[2] += 0.075758f;
                        votes[3] += 0.005682f;
                        votes[4] += 0.894886f;
                        votes[5] += 0.006629f;
                    }
                }
            }
        }
    } else {
        if (features[2] <= -0.142071f) {
            if (features[2] <= -0.254245f) {
                if (features[1] <= 0.260974f) {
                    votes[4] += 1.000000f;
                } else {
                    if (features[0] <= -1.093664f) {
                        votes[0] += 0.066667f;
                        votes[1] += 0.600000f;
                        votes[2] += 0.133333f;
                        votes[3] += 0.066667f;
                        votes[5] += 0.133333f;
                    } else {
                        votes[0] += 0.141463f;
                        votes[1] += 0.151220f;
                        votes[2] += 0.200000f;
                        votes[3] += 0.292683f;
                        votes[4] += 0.019512f;
                        votes[5] += 0.195122f;
                    }
                }
            } else {
                if (features[3] <= 0.879934f) {
                    votes[3] += 1.000000f;
                } else {
                    if (features[1] <= 1.773120f) {
                        votes[0] += 0.212121f;
                        votes[1] += 0.272727f;
                        votes[3] += 0.181818f;
                        votes[5] += 0.333333f;
                    } else {
                        votes[2] += 0.571429f;
                        votes[3] += 0.214286f;
                        votes[4] += 0.214286f;
                    }
                }
            }
        } else {
            if (features[5] <= 0.041135f) {
                if (features[5] <= -0.022824f) {
                    if (features[7] <= 0.034456f) {
                        votes[0] += 0.031376f;
                        votes[1] += 0.228479f;
                        votes[2] += 0.659694f;
                        votes[4] += 0.041432f;
                        votes[5] += 0.039019f;
                    } else {
                        votes[0] += 0.061514f;
                        votes[1] += 0.050473f;
                        votes[2] += 0.067823f;
                        votes[3] += 0.082019f;
                        votes[4] += 0.690852f;
                        votes[5] += 0.047319f;
                    }
                } else {
                    if (features[7] <= 0.293756f) {
                        votes[0] += 0.094039f;
                        votes[1] += 0.644416f;
                        votes[2] += 0.142317f;
                        votes[3] += 0.004198f;
                        votes[4] += 0.011335f;
                        votes[5] += 0.103694f;
                    } else {
                        votes[0] += 0.013746f;
                        votes[1] += 0.027491f;
                        votes[2] += 0.058419f;
                        votes[3] += 0.027491f;
                        votes[4] += 0.859107f;
                        votes[5] += 0.013746f;
                    }
                }
            } else {
                if (features[1] <= -0.242650f) {
                    if (features[5] <= 0.142259f) {
                        votes[0] += 0.257953f;
                        votes[3] += 0.113701f;
                        votes[4] += 0.364724f;
                        votes[5] += 0.263622f;
                    } else {
                        votes[0] += 0.324524f;
                        votes[3] += 0.311606f;
                        votes[4] += 0.022738f;
                        votes[5] += 0.341132f;
                    }
                } else {
                    if (features[1] <= 1.023463f) {
                        votes[0] += 0.012984f;
                        votes[1] += 0.964174f;
                        votes[3] += 0.007694f;
                        votes[4] += 0.004328f;
                        votes[5] += 0.010820f;
                    } else {
                        votes[0] += 0.023491f;
                        votes[1] += 0.019449f;
                        votes[2] += 0.919677f;
                        votes[3] += 0.013892f;
                        votes[4] += 0.007072f;
                        votes[5] += 0.016418f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_1(const float features[8], float votes[6]) {
    if (features[7] <= 0.378691f) {
        if (features[3] <= -0.294329f) {
            if (features[0] <= -0.353816f) {
                if (features[7] <= -6.491477f) {
                    if (features[2] <= -5.903965f) {
                        votes[0] += 0.066667f;
                        votes[1] += 0.091667f;
                        votes[2] += 0.066667f;
                        votes[3] += 0.100000f;
                        votes[4] += 0.600000f;
                        votes[5] += 0.075000f;
                    } else {
                        votes[0] += 0.214286f;
                        votes[1] += 0.185714f;
                        votes[2] += 0.185714f;
                        votes[3] += 0.157143f;
                        votes[4] += 0.028571f;
                        votes[5] += 0.228571f;
                    }
                } else {
                    votes[4] += 1.000000f;
                }
            } else {
                if (features[6] <= 0.973283f) {
                    if (features[5] <= -0.032828f) {
                        votes[0] += 0.027438f;
                        votes[3] += 0.076752f;
                        votes[4] += 0.865777f;
                        votes[5] += 0.030033f;
                    } else {
                        votes[0] += 0.324790f;
                        votes[3] += 0.249235f;
                        votes[4] += 0.101855f;
                        votes[5] += 0.324120f;
                    }
                } else {
                    if (features[0] <= 0.301634f) {
                        votes[0] += 0.469044f;
                        votes[3] += 0.031767f;
                        votes[5] += 0.499190f;
                    } else {
                        votes[0] += 0.271420f;
                        votes[3] += 0.150483f;
                        votes[4] += 0.341146f;
                        votes[5] += 0.236951f;
                    }
                }
            }
        } else {
            if (features[1] <= 0.354382f) {
                if (features[2] <= 0.095337f) {
                    if (features[2] <= -0.355609f) {
                        votes[0] += 0.044944f;
                        votes[1] += 0.011236f;
                        votes[3] += 0.112360f;
                        votes[4] += 0.752809f;
                        votes[5] += 0.078652f;
                    } else {
                        votes[0] += 0.153846f;
                        votes[1] += 0.170940f;
                        votes[3] += 0.512821f;
                        votes[5] += 0.162393f;
                    }
                } else {
                    if (features[1] <= -0.238288f) {
                        votes[0] += 0.242623f;
                        votes[3] += 0.134426f;
                        votes[4] += 0.324590f;
                        votes[5] += 0.298361f;
                    } else {
                        votes[0] += 0.004072f;
                        votes[1] += 0.988111f;
                        votes[3] += 0.000489f;
                        votes[4] += 0.001303f;
                        votes[5] += 0.006026f;
                    }
                }
            } else {
                if (features[3] <= 0.803615f) {
                    if (features[0] <= -0.091990f) {
                        votes[0] += 0.177966f;
                        votes[1] += 0.118644f;
                        votes[2] += 0.097458f;
                        votes[3] += 0.177966f;
                        votes[4] += 0.216102f;
                        votes[5] += 0.211864f;
                    } else {
                        votes[0] += 0.001135f;
                        votes[1] += 0.000810f;
                        votes[2] += 0.996921f;
                        votes[4] += 0.000810f;
                        votes[5] += 0.000324f;
                    }
                } else {
                    if (features[0] <= 0.062044f) {
                        votes[0] += 0.163810f;
                        votes[1] += 0.165405f;
                        votes[2] += 0.172579f;
                        votes[3] += 0.173376f;
                        votes[4] += 0.155839f;
                        votes[5] += 0.168992f;
                    } else {
                        votes[0] += 0.063107f;
                        votes[1] += 0.145631f;
                        votes[2] += 0.567961f;
                        votes[3] += 0.024272f;
                        votes[4] += 0.101942f;
                        votes[5] += 0.097087f;
                    }
                }
            }
        }
    } else {
        if (features[2] <= 0.279331f) {
            if (features[7] <= 9.720194f) {
                if (features[0] <= -0.438603f) {
                    votes[3] += 1.000000f;
                } else {
                    if (features[0] <= -0.324324f) {
                        votes[2] += 1.000000f;
                    } else {
                        votes[0] += 0.178261f;
                        votes[1] += 0.156522f;
                        votes[2] += 0.200000f;
                        votes[3] += 0.134783f;
                        votes[4] += 0.100000f;
                        votes[5] += 0.230435f;
                    }
                }
            } else {
                if (features[4] <= -0.303992f) {
                    if (features[5] <= 4.959652f) {
                        votes[0] += 1.000000f;
                    } else {
                        votes[2] += 1.000000f;
                    }
                } else {
                    votes[4] += 1.000000f;
                }
            }
        } else {
            if (features[3] <= -0.253088f) {
                if (features[6] <= 1.307392f) {
                    if (features[3] <= -0.348413f) {
                        votes[0] += 0.500000f;
                        votes[5] += 0.500000f;
                    } else {
                        votes[3] += 0.998811f;
                        votes[4] += 0.001189f;
                    }
                } else {
                    votes[0] += 1.000000f;
                }
            } else {
                if (features[0] <= 0.690536f) {
                    if (features[5] <= 0.066698f) {
                        votes[0] += 0.055556f;
                        votes[1] += 0.111111f;
                        votes[3] += 0.305556f;
                        votes[4] += 0.500000f;
                        votes[5] += 0.027778f;
                    } else {
                        votes[0] += 0.231939f;
                        votes[1] += 0.201521f;
                        votes[2] += 0.167300f;
                        votes[3] += 0.174905f;
                        votes[4] += 0.068441f;
                        votes[5] += 0.155894f;
                    }
                } else {
                    if (features[5] <= 1.013076f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.125000f;
                        votes[3] += 0.625000f;
                        votes[5] += 0.250000f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_2(const float features[8], float votes[6]) {
    if (features[6] <= 0.966681f) {
        if (features[5] <= -0.116263f) {
            if (features[2] <= -0.585580f) {
                if (features[0] <= -5.903850f) {
                    if (features[4] <= -2.783397f) {
                        votes[0] += 0.086538f;
                        votes[1] += 0.096154f;
                        votes[2] += 0.076923f;
                        votes[3] += 0.096154f;
                        votes[4] += 0.557692f;
                        votes[5] += 0.086538f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[2] <= -0.699101f) {
                        votes[0] += 0.171605f;
                        votes[1] += 0.166667f;
                        votes[2] += 0.160905f;
                        votes[3] += 0.165021f;
                        votes[4] += 0.159259f;
                        votes[5] += 0.176543f;
                    } else {
                        votes[0] += 0.159091f;
                        votes[1] += 0.022727f;
                        votes[2] += 0.454545f;
                        votes[3] += 0.113636f;
                        votes[4] += 0.136364f;
                        votes[5] += 0.113636f;
                    }
                }
            } else {
                if (features[5] <= -0.174119f) {
                    if (features[2] <= 0.301007f) {
                        votes[0] += 0.014174f;
                        votes[1] += 0.012048f;
                        votes[2] += 0.063785f;
                        votes[3] += 0.014883f;
                        votes[4] += 0.879518f;
                        votes[5] += 0.015592f;
                    } else {
                        votes[1] += 0.001432f;
                        votes[4] += 0.998568f;
                    }
                } else {
                    if (features[1] <= 0.509543f) {
                        votes[0] += 0.030894f;
                        votes[1] += 0.047154f;
                        votes[3] += 0.125203f;
                        votes[4] += 0.760976f;
                        votes[5] += 0.035772f;
                    } else {
                        votes[2] += 0.981618f;
                        votes[3] += 0.007353f;
                        votes[4] += 0.011029f;
                    }
                }
            }
        } else {
            if (features[1] <= -0.226738f) {
                if (features[3] <= -0.339548f) {
                    if (features[6] <= 0.734942f) {
                        votes[0] += 0.384869f;
                        votes[3] += 0.203865f;
                        votes[4] += 0.032878f;
                        votes[5] += 0.378388f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                } else {
                    if (features[6] <= -0.876325f) {
                        votes[0] += 0.006309f;
                        votes[3] += 0.993691f;
                    } else {
                        votes[0] += 0.069343f;
                        votes[3] += 0.464416f;
                        votes[4] += 0.397202f;
                        votes[5] += 0.069039f;
                    }
                }
            } else {
                if (features[0] <= 0.304686f) {
                    if (features[2] <= 0.300057f) {
                        votes[0] += 0.029046f;
                        votes[1] += 0.231443f;
                        votes[2] += 0.652835f;
                        votes[3] += 0.043338f;
                        votes[4] += 0.014292f;
                        votes[5] += 0.029046f;
                    } else {
                        votes[1] += 0.800485f;
                        votes[2] += 0.199515f;
                    }
                } else {
                    if (features[0] <= 0.325575f) {
                        votes[2] += 0.999012f;
                        votes[3] += 0.000988f;
                    } else {
                        votes[0] += 0.156951f;
                        votes[1] += 0.165919f;
                        votes[2] += 0.188341f;
                        votes[3] += 0.188341f;
                        votes[4] += 0.116592f;
                        votes[5] += 0.183857f;
                    }
                }
            }
        }
    } else {
        if (features[3] <= 0.574433f) {
            if (features[1] <= -0.232618f) {
                if (features[7] <= 0.431272f) {
                    if (features[4] <= 1.118503f) {
                        votes[0] += 0.475071f;
                        votes[3] += 0.062559f;
                        votes[4] += 0.011759f;
                        votes[5] += 0.450611f;
                    } else {
                        votes[0] += 0.301530f;
                        votes[3] += 0.121731f;
                        votes[4] += 0.279322f;
                        votes[5] += 0.297417f;
                    }
                } else {
                    if (features[2] <= 0.249393f) {
                        votes[0] += 0.428571f;
                        votes[5] += 0.571429f;
                    } else {
                        votes[0] += 0.001092f;
                        votes[3] += 0.996725f;
                        votes[5] += 0.002183f;
                    }
                }
            } else {
                if (features[6] <= 1.317138f) {
                    if (features[7] <= 0.363472f) {
                        votes[0] += 0.008276f;
                        votes[1] += 0.986326f;
                        votes[3] += 0.001080f;
                        votes[5] += 0.004318f;
                    } else {
                        votes[0] += 0.261905f;
                        votes[1] += 0.452381f;
                        votes[3] += 0.095238f;
                        votes[5] += 0.190476f;
                    }
                } else {
                    if (features[6] <= 1.327488f) {
                        votes[5] += 1.000000f;
                    } else {
                        votes[1] += 0.666667f;
                        votes[4] += 0.333333f;
                    }
                }
            }
        } else {
            if (features[1] <= 1.315731f) {
                if (features[3] <= 1.025006f) {
                    if (features[7] <= 0.510055f) {
                        votes[1] += 0.001786f;
                        votes[2] += 0.995355f;
                        votes[3] += 0.002501f;
                        votes[5] += 0.000357f;
                    } else {
                        votes[0] += 0.555556f;
                        votes[5] += 0.444444f;
                    }
                } else {
                    if (features[2] <= -0.252837f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.297297f;
                        votes[1] += 0.162162f;
                        votes[2] += 0.121622f;
                        votes[3] += 0.054054f;
                        votes[4] += 0.121622f;
                        votes[5] += 0.243243f;
                    }
                }
            } else {
                if (features[3] <= 1.595383f) {
                    if (features[6] <= 1.345229f) {
                        votes[0] += 0.071429f;
                        votes[1] += 0.071429f;
                        votes[2] += 0.803571f;
                        votes[3] += 0.017857f;
                        votes[5] += 0.035714f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[7] <= 0.669098f) {
                        votes[0] += 0.221477f;
                        votes[1] += 0.234899f;
                        votes[2] += 0.255034f;
                        votes[3] += 0.033557f;
                        votes[4] += 0.033557f;
                        votes[5] += 0.221477f;
                    } else {
                        votes[1] += 0.281250f;
                        votes[2] += 0.156250f;
                        votes[3] += 0.281250f;
                        votes[4] += 0.250000f;
                        votes[5] += 0.031250f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_3(const float features[8], float votes[6]) {
    if (features[5] <= -0.095281f) {
        if (features[6] <= -0.882435f) {
            if (features[7] <= -6.792832f) {
                if (features[4] <= -2.783397f) {
                    votes[0] += 0.080000f;
                    votes[1] += 0.080000f;
                    votes[2] += 0.053333f;
                    votes[3] += 0.106667f;
                    votes[4] += 0.586667f;
                    votes[5] += 0.093333f;
                } else {
                    votes[4] += 1.000000f;
                }
            } else {
                if (features[1] <= 1.124443f) {
                    if (features[0] <= -4.593925f) {
                        votes[0] += 0.137363f;
                        votes[1] += 0.167582f;
                        votes[2] += 0.186813f;
                        votes[3] += 0.186813f;
                        votes[4] += 0.162088f;
                        votes[5] += 0.159341f;
                    } else {
                        votes[0] += 0.165354f;
                        votes[1] += 0.086614f;
                        votes[2] += 0.031496f;
                        votes[3] += 0.480315f;
                        votes[4] += 0.149606f;
                        votes[5] += 0.086614f;
                    }
                } else {
                    if (features[2] <= -1.506110f) {
                        votes[0] += 0.175936f;
                        votes[1] += 0.163451f;
                        votes[2] += 0.173666f;
                        votes[3] += 0.113507f;
                        votes[4] += 0.189557f;
                        votes[5] += 0.183882f;
                    } else {
                        votes[0] += 0.034483f;
                        votes[1] += 0.068966f;
                        votes[2] += 0.068966f;
                        votes[3] += 0.103448f;
                        votes[4] += 0.620690f;
                        votes[5] += 0.103448f;
                    }
                }
            }
        } else {
            if (features[1] <= -0.227042f) {
                if (features[5] <= -0.169043f) {
                    if (features[7] <= 0.060639f) {
                        votes[0] += 0.003385f;
                        votes[4] += 0.992308f;
                        votes[5] += 0.004308f;
                    } else {
                        votes[0] += 0.025000f;
                        votes[3] += 0.193750f;
                        votes[4] += 0.743750f;
                        votes[5] += 0.037500f;
                    }
                } else {
                    if (features[2] <= 0.300630f) {
                        votes[0] += 0.337838f;
                        votes[3] += 0.229730f;
                        votes[4] += 0.088803f;
                        votes[5] += 0.343629f;
                    } else {
                        votes[0] += 0.008963f;
                        votes[4] += 0.985915f;
                        votes[5] += 0.005122f;
                    }
                }
            } else {
                if (features[1] <= -0.025059f) {
                    if (features[7] <= -0.128786f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[1] += 0.971519f;
                        votes[3] += 0.003165f;
                        votes[4] += 0.025316f;
                    }
                } else {
                    if (features[1] <= 1.071509f) {
                        votes[0] += 0.022016f;
                        votes[1] += 0.015064f;
                        votes[2] += 0.819235f;
                        votes[3] += 0.064890f;
                        votes[4] += 0.056779f;
                        votes[5] += 0.022016f;
                    } else {
                        votes[0] += 0.161661f;
                        votes[1] += 0.166078f;
                        votes[2] += 0.205830f;
                        votes[3] += 0.113958f;
                        votes[4] += 0.193463f;
                        votes[5] += 0.159011f;
                    }
                }
            }
        }
    } else {
        if (features[1] <= -0.226514f) {
            if (features[7] <= 0.377955f) {
                if (features[3] <= -0.339649f) {
                    if (features[5] <= 0.362872f) {
                        votes[0] += 0.326311f;
                        votes[3] += 0.321152f;
                        votes[4] += 0.025150f;
                        votes[5] += 0.327386f;
                    } else {
                        votes[0] += 0.460018f;
                        votes[3] += 0.081745f;
                        votes[5] += 0.458237f;
                    }
                } else {
                    if (features[5] <= 0.224338f) {
                        votes[0] += 0.031426f;
                        votes[3] += 0.015947f;
                        votes[4] += 0.917448f;
                        votes[5] += 0.035178f;
                    } else {
                        votes[0] += 0.443709f;
                        votes[3] += 0.079470f;
                        votes[4] += 0.084768f;
                        votes[5] += 0.392053f;
                    }
                }
            } else {
                if (features[2] <= 0.260881f) {
                    if (features[6] <= -0.822024f) {
                        votes[0] += 0.187500f;
                        votes[3] += 0.500000f;
                        votes[4] += 0.125000f;
                        votes[5] += 0.187500f;
                    } else {
                        votes[0] += 0.580645f;
                        votes[3] += 0.161290f;
                        votes[5] += 0.258065f;
                    }
                } else {
                    if (features[3] <= -0.297885f) {
                        votes[3] += 0.998049f;
                        votes[4] += 0.001171f;
                        votes[5] += 0.000780f;
                    } else {
                        votes[0] += 0.090909f;
                        votes[3] += 0.500000f;
                        votes[5] += 0.409091f;
                    }
                }
            }
        } else {
            if (features[5] <= 0.298404f) {
                if (features[0] <= 0.304683f) {
                    if (features[0] <= 0.300045f) {
                        votes[0] += 0.026976f;
                        votes[1] += 0.217709f;
                        votes[2] += 0.709616f;
                        votes[3] += 0.017772f;
                        votes[4] += 0.004126f;
                        votes[5] += 0.023802f;
                    } else {
                        votes[1] += 0.732050f;
                        votes[2] += 0.267950f;
                    }
                } else {
                    if (features[7] <= 0.034381f) {
                        votes[2] += 1.000000f;
                    } else {
                        votes[0] += 0.010919f;
                        votes[1] += 0.218380f;
                        votes[2] += 0.729754f;
                        votes[3] += 0.012739f;
                        votes[4] += 0.019108f;
                        votes[5] += 0.009099f;
                    }
                }
            } else {
                if (features[3] <= 0.538134f) {
                    if (features[1] <= -0.166311f) {
                        votes[0] += 0.090909f;
                        votes[3] += 0.545455f;
                        votes[5] += 0.363636f;
                    } else {
                        votes[0] += 0.005190f;
                        votes[1] += 0.986028f;
                        votes[3] += 0.005190f;
                        votes[5] += 0.003593f;
                    }
                } else {
                    if (features[7] <= 0.371852f) {
                        votes[0] += 0.035792f;
                        votes[1] += 0.021692f;
                        votes[2] += 0.869848f;
                        votes[3] += 0.049892f;
                        votes[5] += 0.022777f;
                    } else {
                        votes[0] += 0.169697f;
                        votes[1] += 0.233333f;
                        votes[2] += 0.218182f;
                        votes[3] += 0.130303f;
                        votes[4] += 0.057576f;
                        votes[5] += 0.190909f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_4(const float features[8], float votes[6]) {
    if (features[7] <= 0.378691f) {
        if (features[1] <= -0.225320f) {
            if (features[5] <= -0.022825f) {
                if (features[7] <= -0.014051f) {
                    if (features[4] <= 0.765454f) {
                        votes[0] += 0.028728f;
                        votes[1] += 0.027360f;
                        votes[2] += 0.019152f;
                        votes[3] += 0.027360f;
                        votes[4] += 0.865937f;
                        votes[5] += 0.031464f;
                    } else {
                        votes[0] += 0.320000f;
                        votes[3] += 0.329524f;
                        votes[4] += 0.003810f;
                        votes[5] += 0.346667f;
                    }
                } else {
                    if (features[2] <= 0.301304f) {
                        votes[0] += 0.073413f;
                        votes[3] += 0.057540f;
                        votes[4] += 0.777778f;
                        votes[5] += 0.091270f;
                    } else {
                        votes[0] += 0.007337f;
                        votes[3] += 0.004891f;
                        votes[4] += 0.976460f;
                        votes[5] += 0.011312f;
                    }
                }
            } else {
                if (features[7] <= 0.285006f) {
                    if (features[0] <= -0.038121f) {
                        votes[3] += 0.908571f;
                        votes[4] += 0.091429f;
                    } else {
                        votes[0] += 0.352096f;
                        votes[3] += 0.296171f;
                        votes[4] += 0.004469f;
                        votes[5] += 0.347264f;
                    }
                } else {
                    if (features[5] <= 0.153523f) {
                        votes[0] += 0.007452f;
                        votes[3] += 0.002981f;
                        votes[4] += 0.981371f;
                        votes[5] += 0.008197f;
                    } else {
                        votes[0] += 0.469256f;
                        votes[3] += 0.016917f;
                        votes[4] += 0.045749f;
                        votes[5] += 0.468079f;
                    }
                }
            }
        } else {
            if (features[1] <= 0.354382f) {
                if (features[5] <= -0.233975f) {
                    if (features[3] <= -0.145805f) {
                        votes[1] += 1.000000f;
                    } else {
                        votes[0] += 0.175439f;
                        votes[1] += 0.052632f;
                        votes[2] += 0.070175f;
                        votes[3] += 0.140351f;
                        votes[4] += 0.473684f;
                        votes[5] += 0.087719f;
                    }
                } else {
                    if (features[6] <= 1.317138f) {
                        votes[0] += 0.004797f;
                        votes[1] += 0.984650f;
                        votes[3] += 0.003678f;
                        votes[4] += 0.001119f;
                        votes[5] += 0.005756f;
                    } else {
                        votes[0] += 0.750000f;
                        votes[5] += 0.250000f;
                    }
                }
            } else {
                if (features[0] <= 0.051992f) {
                    if (features[6] <= 0.881376f) {
                        votes[0] += 0.168964f;
                        votes[1] += 0.155179f;
                        votes[2] += 0.176447f;
                        votes[3] += 0.168570f;
                        votes[4] += 0.167783f;
                        votes[5] += 0.163056f;
                    } else {
                        votes[0] += 0.290323f;
                        votes[1] += 0.134409f;
                        votes[2] += 0.204301f;
                        votes[3] += 0.123656f;
                        votes[5] += 0.247312f;
                    }
                } else {
                    if (features[6] <= -0.903943f) {
                        votes[0] += 0.230769f;
                        votes[1] += 0.096154f;
                        votes[2] += 0.173077f;
                        votes[3] += 0.076923f;
                        votes[4] += 0.250000f;
                        votes[5] += 0.173077f;
                    } else {
                        votes[0] += 0.003043f;
                        votes[1] += 0.004164f;
                        votes[2] += 0.986387f;
                        votes[3] += 0.001441f;
                        votes[4] += 0.002082f;
                        votes[5] += 0.002883f;
                    }
                }
            }
        }
    } else {
        if (features[5] <= 0.761569f) {
            if (features[3] <= -0.312650f) {
                if (features[2] <= 0.338948f) {
                    if (features[6] <= -0.883992f) {
                        votes[3] += 0.977143f;
                        votes[4] += 0.022857f;
                    } else {
                        votes[0] += 0.000411f;
                        votes[3] += 0.998356f;
                        votes[4] += 0.000822f;
                        votes[5] += 0.000411f;
                    }
                } else {
                    votes[0] += 0.428571f;
                    votes[5] += 0.571429f;
                }
            } else {
                if (features[5] <= 0.066698f) {
                    if (features[5] <= -0.986536f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.030769f;
                        votes[1] += 0.076923f;
                        votes[2] += 0.107692f;
                        votes[3] += 0.046154f;
                        votes[4] += 0.707692f;
                        votes[5] += 0.030769f;
                    }
                } else {
                    if (features[3] <= 0.230088f) {
                        votes[0] += 0.184211f;
                        votes[1] += 0.149123f;
                        votes[3] += 0.438596f;
                        votes[4] += 0.061404f;
                        votes[5] += 0.166667f;
                    } else {
                        votes[0] += 0.180095f;
                        votes[1] += 0.170616f;
                        votes[2] += 0.270142f;
                        votes[3] += 0.113744f;
                        votes[4] += 0.052133f;
                        votes[5] += 0.213270f;
                    }
                }
            }
        } else {
            if (features[2] <= 0.434501f) {
                if (features[0] <= 0.268884f) {
                    if (features[7] <= 6.509367f) {
                        votes[0] += 0.208333f;
                        votes[1] += 0.041667f;
                        votes[2] += 0.180556f;
                        votes[3] += 0.236111f;
                        votes[4] += 0.041667f;
                        votes[5] += 0.291667f;
                    } else {
                        votes[0] += 0.055556f;
                        votes[1] += 0.388889f;
                        votes[2] += 0.222222f;
                        votes[4] += 0.166667f;
                        votes[5] += 0.166667f;
                    }
                } else {
                    if (features[5] <= 0.784036f) {
                        votes[0] += 0.750000f;
                        votes[5] += 0.250000f;
                    } else {
                        votes[1] += 0.021739f;
                        votes[2] += 0.130435f;
                        votes[3] += 0.717391f;
                        votes[4] += 0.108696f;
                        votes[5] += 0.021739f;
                    }
                }
            } else {
                if (features[3] <= 0.505881f) {
                    votes[3] += 1.000000f;
                } else {
                    if (features[3] <= 1.119927f) {
                        votes[0] += 0.333333f;
                        votes[1] += 0.416667f;
                        votes[3] += 0.083333f;
                        votes[5] += 0.166667f;
                    } else {
                        votes[0] += 0.278481f;
                        votes[1] += 0.113924f;
                        votes[2] += 0.189873f;
                        votes[3] += 0.037975f;
                        votes[4] += 0.088608f;
                        votes[5] += 0.291139f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_5(const float features[8], float votes[6]) {
    if (features[1] <= -0.225320f) {
        if (features[0] <= -0.356363f) {
            if (features[0] <= -3.230669f) {
                if (features[4] <= -2.781550f) {
                    if (features[3] <= -1.326298f) {
                        votes[0] += 0.077670f;
                        votes[1] += 0.067961f;
                        votes[2] += 0.077670f;
                        votes[3] += 0.087379f;
                        votes[4] += 0.611650f;
                        votes[5] += 0.077670f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[4] <= -2.693520f) {
                        votes[0] += 0.183333f;
                        votes[1] += 0.133333f;
                        votes[2] += 0.133333f;
                        votes[3] += 0.133333f;
                        votes[4] += 0.266667f;
                        votes[5] += 0.150000f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                }
            } else {
                votes[4] += 1.000000f;
            }
        } else {
            if (features[5] <= -0.022825f) {
                if (features[0] <= 0.120550f) {
                    if (features[4] <= 0.972769f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.509589f;
                        votes[5] += 0.490411f;
                    }
                } else {
                    if (features[5] <= -0.089175f) {
                        votes[0] += 0.015308f;
                        votes[3] += 0.012953f;
                        votes[4] += 0.956727f;
                        votes[5] += 0.015013f;
                    } else {
                        votes[0] += 0.119919f;
                        votes[3] += 0.060976f;
                        votes[4] += 0.725610f;
                        votes[5] += 0.093496f;
                    }
                }
            } else {
                if (features[7] <= 0.370586f) {
                    if (features[3] <= -0.339668f) {
                        votes[0] += 0.378278f;
                        votes[3] += 0.233485f;
                        votes[4] += 0.003862f;
                        votes[5] += 0.384376f;
                    } else {
                        votes[0] += 0.149369f;
                        votes[1] += 0.000814f;
                        votes[3] += 0.035409f;
                        votes[4] += 0.675621f;
                        votes[5] += 0.138787f;
                    }
                } else {
                    if (features[3] <= -0.312934f) {
                        votes[0] += 0.000789f;
                        votes[3] += 0.995266f;
                        votes[4] += 0.001972f;
                        votes[5] += 0.001972f;
                    } else {
                        votes[0] += 0.186992f;
                        votes[3] += 0.626016f;
                        votes[4] += 0.048780f;
                        votes[5] += 0.138211f;
                    }
                }
            }
        }
    } else {
        if (features[1] <= 0.375006f) {
            if (features[0] <= -0.069581f) {
                if (features[4] <= -1.045727f) {
                    if (features[0] <= -5.843069f) {
                        votes[0] += 0.151515f;
                        votes[1] += 0.121212f;
                        votes[2] += 0.030303f;
                        votes[3] += 0.272727f;
                        votes[4] += 0.121212f;
                        votes[5] += 0.303030f;
                    } else {
                        votes[1] += 0.500000f;
                        votes[2] += 0.500000f;
                    }
                } else {
                    if (features[0] <= -0.357237f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[3] += 0.807692f;
                        votes[4] += 0.192308f;
                    }
                }
            } else {
                if (features[7] <= 0.363472f) {
                    if (features[0] <= 0.101900f) {
                        votes[0] += 0.140000f;
                        votes[1] += 0.740000f;
                        votes[5] += 0.120000f;
                    } else {
                        votes[0] += 0.004271f;
                        votes[1] += 0.991786f;
                        votes[3] += 0.000164f;
                        votes[4] += 0.000986f;
                        votes[5] += 0.002793f;
                    }
                } else {
                    if (features[5] <= 0.200887f) {
                        votes[0] += 0.333333f;
                        votes[4] += 0.333333f;
                        votes[5] += 0.333333f;
                    } else {
                        votes[0] += 0.121212f;
                        votes[1] += 0.575758f;
                        votes[3] += 0.136364f;
                        votes[4] += 0.015152f;
                        votes[5] += 0.151515f;
                    }
                }
            }
        } else {
            if (features[3] <= 0.796469f) {
                if (features[5] <= -0.678095f) {
                    if (features[2] <= -1.038809f) {
                        votes[0] += 0.173684f;
                        votes[1] += 0.131579f;
                        votes[2] += 0.073684f;
                        votes[3] += 0.294737f;
                        votes[4] += 0.131579f;
                        votes[5] += 0.194737f;
                    } else {
                        votes[2] += 0.375000f;
                        votes[4] += 0.625000f;
                    }
                } else {
                    if (features[4] <= -0.918138f) {
                        votes[0] += 0.090909f;
                        votes[1] += 0.363636f;
                        votes[4] += 0.181818f;
                        votes[5] += 0.363636f;
                    } else {
                        votes[0] += 0.003048f;
                        votes[1] += 0.000321f;
                        votes[2] += 0.991818f;
                        votes[3] += 0.002407f;
                        votes[4] += 0.000321f;
                        votes[5] += 0.002086f;
                    }
                }
            } else {
                if (features[3] <= 2.374694f) {
                    if (features[4] <= 0.886843f) {
                        votes[0] += 0.158267f;
                        votes[1] += 0.171088f;
                        votes[2] += 0.169319f;
                        votes[3] += 0.175066f;
                        votes[4] += 0.157825f;
                        votes[5] += 0.168435f;
                    } else {
                        votes[0] += 0.179487f;
                        votes[1] += 0.198718f;
                        votes[2] += 0.310897f;
                        votes[3] += 0.125000f;
                        votes[4] += 0.009615f;
                        votes[5] += 0.176282f;
                    }
                } else {
                    if (features[3] <= 3.172662f) {
                        votes[0] += 0.055814f;
                        votes[1] += 0.116279f;
                        votes[2] += 0.595349f;
                        votes[3] += 0.065116f;
                        votes[4] += 0.065116f;
                        votes[5] += 0.102326f;
                    } else {
                        votes[0] += 0.206250f;
                        votes[1] += 0.121875f;
                        votes[2] += 0.200000f;
                        votes[3] += 0.140625f;
                        votes[4] += 0.146875f;
                        votes[5] += 0.184375f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_6(const float features[8], float votes[6]) {
    if (features[2] <= 0.304738f) {
        if (features[3] <= -0.294107f) {
            if (features[7] <= 0.363812f) {
                if (features[5] <= -0.040292f) {
                    if (features[5] <= -0.170007f) {
                        votes[0] += 0.013509f;
                        votes[1] += 0.005969f;
                        votes[2] += 0.005341f;
                        votes[3] += 0.011938f;
                        votes[4] += 0.950675f;
                        votes[5] += 0.012567f;
                    } else {
                        votes[0] += 0.141296f;
                        votes[3] += 0.121780f;
                        votes[4] += 0.574551f;
                        votes[5] += 0.162373f;
                    }
                } else {
                    if (features[3] <= -0.339684f) {
                        votes[0] += 0.378978f;
                        votes[3] += 0.225038f;
                        votes[4] += 0.009835f;
                        votes[5] += 0.386149f;
                    } else {
                        votes[0] += 0.105444f;
                        votes[3] += 0.019127f;
                        votes[4] += 0.772437f;
                        votes[5] += 0.102992f;
                    }
                }
            } else {
                if (features[3] <= -0.329261f) {
                    if (features[4] <= -0.887674f) {
                        votes[3] += 0.994220f;
                        votes[4] += 0.005780f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                } else {
                    if (features[2] <= 0.252393f) {
                        votes[0] += 0.615385f;
                        votes[3] += 0.076923f;
                        votes[4] += 0.076923f;
                        votes[5] += 0.230769f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                }
            }
        } else {
            if (features[5] <= -0.167352f) {
                if (features[1] <= 0.824805f) {
                    if (features[5] <= -0.852690f) {
                        votes[0] += 0.087500f;
                        votes[1] += 0.150000f;
                        votes[2] += 0.125000f;
                        votes[3] += 0.200000f;
                        votes[4] += 0.337500f;
                        votes[5] += 0.100000f;
                    } else {
                        votes[1] += 0.130769f;
                        votes[4] += 0.869231f;
                    }
                } else {
                    if (features[5] <= -0.586317f) {
                        votes[0] += 0.166273f;
                        votes[1] += 0.154464f;
                        votes[2] += 0.160132f;
                        votes[3] += 0.161077f;
                        votes[4] += 0.189419f;
                        votes[5] += 0.168635f;
                    } else {
                        votes[0] += 0.126464f;
                        votes[1] += 0.098361f;
                        votes[2] += 0.379391f;
                        votes[3] += 0.154567f;
                        votes[4] += 0.119438f;
                        votes[5] += 0.121780f;
                    }
                }
            } else {
                if (features[2] <= 0.300057f) {
                    if (features[1] <= 0.917304f) {
                        votes[0] += 0.057477f;
                        votes[1] += 0.768566f;
                        votes[3] += 0.073754f;
                        votes[4] += 0.033062f;
                        votes[5] += 0.067141f;
                    } else {
                        votes[0] += 0.026338f;
                        votes[1] += 0.032637f;
                        votes[2] += 0.878614f;
                        votes[3] += 0.030633f;
                        votes[4] += 0.006585f;
                        votes[5] += 0.025193f;
                    }
                } else {
                    if (features[7] <= 0.015920f) {
                        votes[1] += 0.181063f;
                        votes[2] += 0.818937f;
                    } else {
                        votes[1] += 0.821736f;
                        votes[2] += 0.178264f;
                    }
                }
            }
        }
    } else {
        if (features[3] <= 0.585575f) {
            if (features[2] <= 0.305057f) {
                if (features[5] <= -0.004481f) {
                    if (features[1] <= -0.302563f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[1] += 1.000000f;
                    }
                } else {
                    if (features[2] <= 0.304815f) {
                        votes[1] += 0.929293f;
                        votes[3] += 0.020202f;
                        votes[5] += 0.050505f;
                    } else {
                        votes[0] += 0.092308f;
                        votes[1] += 0.415385f;
                        votes[3] += 0.246154f;
                        votes[4] += 0.123077f;
                        votes[5] += 0.123077f;
                    }
                }
            } else {
                if (features[1] <= -0.230551f) {
                    if (features[6] <= 1.286161f) {
                        votes[0] += 0.266854f;
                        votes[3] += 0.176966f;
                        votes[4] += 0.353933f;
                        votes[5] += 0.202247f;
                    } else {
                        votes[0] += 0.343066f;
                        votes[3] += 0.182482f;
                        votes[4] += 0.094891f;
                        votes[5] += 0.379562f;
                    }
                } else {
                    if (features[0] <= 0.432667f) {
                        votes[1] += 0.972332f;
                        votes[3] += 0.011858f;
                        votes[4] += 0.015810f;
                    } else {
                        votes[0] += 0.571429f;
                        votes[3] += 0.285714f;
                        votes[5] += 0.142857f;
                    }
                }
            }
        } else {
            if (features[7] <= 0.408827f) {
                if (features[3] <= 0.965108f) {
                    votes[2] += 1.000000f;
                } else {
                    if (features[2] <= 0.339588f) {
                        votes[2] += 0.230769f;
                        votes[4] += 0.769231f;
                    } else {
                        votes[0] += 0.062500f;
                        votes[1] += 0.437500f;
                        votes[2] += 0.187500f;
                        votes[3] += 0.187500f;
                        votes[5] += 0.125000f;
                    }
                }
            } else {
                if (features[1] <= 13.605051f) {
                    if (features[3] <= 6.284412f) {
                        votes[0] += 0.162679f;
                        votes[1] += 0.124402f;
                        votes[2] += 0.229665f;
                        votes[3] += 0.191388f;
                        votes[4] += 0.119617f;
                        votes[5] += 0.172249f;
                    } else {
                        votes[2] += 0.285714f;
                        votes[3] += 0.178571f;
                        votes[4] += 0.535714f;
                    }
                } else {
                    if (features[6] <= 0.792732f) {
                        votes[0] += 0.346154f;
                        votes[1] += 0.038462f;
                        votes[2] += 0.076923f;
                        votes[5] += 0.538462f;
                    } else {
                        votes[1] += 1.000000f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_7(const float features[8], float votes[6]) {
    if (features[3] <= -0.294107f) {
        if (features[3] <= -0.334759f) {
            if (features[3] <= -0.339576f) {
                if (features[3] <= -0.349884f) {
                    if (features[1] <= -0.504892f) {
                        votes[0] += 0.300214f;
                        votes[1] += 0.002449f;
                        votes[2] += 0.004133f;
                        votes[3] += 0.182792f;
                        votes[4] += 0.208053f;
                        votes[5] += 0.302358f;
                    } else {
                        votes[0] += 0.169376f;
                        votes[3] += 0.345138f;
                        votes[4] += 0.321626f;
                        votes[5] += 0.163861f;
                    }
                } else {
                    if (features[0] <= -0.060623f) {
                        votes[3] += 0.910000f;
                        votes[4] += 0.090000f;
                    } else {
                        votes[0] += 0.493303f;
                        votes[3] += 0.002435f;
                        votes[4] += 0.009741f;
                        votes[5] += 0.494521f;
                    }
                }
            } else {
                if (features[5] <= 0.280595f) {
                    if (features[1] <= -0.397298f) {
                        votes[0] += 0.000428f;
                        votes[4] += 0.999144f;
                        votes[5] += 0.000428f;
                    } else {
                        votes[0] += 0.403509f;
                        votes[4] += 0.140351f;
                        votes[5] += 0.456140f;
                    }
                } else {
                    if (features[5] <= 0.582546f) {
                        votes[0] += 0.494118f;
                        votes[3] += 0.047059f;
                        votes[4] += 0.017647f;
                        votes[5] += 0.441176f;
                    } else {
                        votes[0] += 0.205882f;
                        votes[3] += 0.676471f;
                        votes[5] += 0.117647f;
                    }
                }
            }
        } else {
            if (features[0] <= 0.297988f) {
                if (features[4] <= 1.001512f) {
                    if (features[6] <= 0.759002f) {
                        votes[0] += 0.272109f;
                        votes[3] += 0.224490f;
                        votes[4] += 0.267574f;
                        votes[5] += 0.235828f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                } else {
                    if (features[1] <= -0.357261f) {
                        votes[0] += 0.477778f;
                        votes[5] += 0.522222f;
                    } else {
                        votes[0] += 0.777778f;
                        votes[5] += 0.222222f;
                    }
                }
            } else {
                if (features[6] <= 1.130336f) {
                    if (features[5] <= -0.105635f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.002066f;
                        votes[3] += 0.995351f;
                        votes[4] += 0.001033f;
                        votes[5] += 0.001550f;
                    }
                } else {
                    if (features[5] <= 0.448729f) {
                        votes[0] += 0.168067f;
                        votes[1] += 0.025210f;
                        votes[3] += 0.075630f;
                        votes[4] += 0.521008f;
                        votes[5] += 0.210084f;
                    } else {
                        votes[0] += 0.045249f;
                        votes[3] += 0.915535f;
                        votes[5] += 0.039216f;
                    }
                }
            }
        }
    } else {
        if (features[2] <= 0.095130f) {
            if (features[5] <= -0.175685f) {
                if (features[2] <= -0.587089f) {
                    if (features[6] <= 0.299464f) {
                        votes[0] += 0.176121f;
                        votes[1] += 0.163769f;
                        votes[2] += 0.170631f;
                        votes[3] += 0.176578f;
                        votes[4] += 0.156908f;
                        votes[5] += 0.155993f;
                    } else {
                        votes[0] += 0.203463f;
                        votes[1] += 0.138528f;
                        votes[2] += 0.155844f;
                        votes[3] += 0.047619f;
                        votes[4] += 0.242424f;
                        votes[5] += 0.212121f;
                    }
                } else {
                    if (features[6] <= 0.886777f) {
                        votes[1] += 0.066667f;
                        votes[3] += 0.053333f;
                        votes[4] += 0.880000f;
                    } else {
                        votes[0] += 0.333333f;
                        votes[1] += 0.277778f;
                        votes[2] += 0.111111f;
                        votes[5] += 0.277778f;
                    }
                }
            } else {
                if (features[1] <= 1.655661f) {
                    if (features[1] <= 1.305106f) {
                        votes[0] += 0.125698f;
                        votes[1] += 0.162011f;
                        votes[2] += 0.097765f;
                        votes[3] += 0.435754f;
                        votes[4] += 0.050279f;
                        votes[5] += 0.128492f;
                    } else {
                        votes[0] += 0.309524f;
                        votes[1] += 0.303571f;
                        votes[2] += 0.041667f;
                        votes[3] += 0.011905f;
                        votes[4] += 0.017857f;
                        votes[5] += 0.315476f;
                    }
                } else {
                    if (features[0] <= -0.225372f) {
                        votes[2] += 1.000000f;
                    } else {
                        votes[0] += 0.103896f;
                        votes[1] += 0.129870f;
                        votes[2] += 0.311688f;
                        votes[3] += 0.324675f;
                        votes[4] += 0.103896f;
                        votes[5] += 0.025974f;
                    }
                }
            }
        } else {
            if (features[1] <= 0.830332f) {
                if (features[7] <= 0.363128f) {
                    if (features[4] <= 1.311930f) {
                        votes[0] += 0.015526f;
                        votes[1] += 0.950474f;
                        votes[2] += 0.000311f;
                        votes[3] += 0.001708f;
                        votes[4] += 0.016923f;
                        votes[5] += 0.015060f;
                    } else {
                        votes[0] += 0.294118f;
                        votes[3] += 0.058824f;
                        votes[4] += 0.235294f;
                        votes[5] += 0.411765f;
                    }
                } else {
                    if (features[0] <= 0.153294f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.220339f;
                        votes[1] += 0.173729f;
                        votes[3] += 0.330508f;
                        votes[4] += 0.097458f;
                        votes[5] += 0.177966f;
                    }
                }
            } else {
                if (features[4] <= -0.890784f) {
                    if (features[5] <= -0.302382f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.193370f;
                        votes[1] += 0.154696f;
                        votes[2] += 0.171271f;
                        votes[3] += 0.160221f;
                        votes[4] += 0.138122f;
                        votes[5] += 0.182320f;
                    }
                } else {
                    if (features[3] <= 1.298823f) {
                        votes[0] += 0.000332f;
                        votes[2] += 0.998670f;
                        votes[4] += 0.000332f;
                        votes[5] += 0.000665f;
                    } else {
                        votes[0] += 0.110048f;
                        votes[1] += 0.196172f;
                        votes[2] += 0.287081f;
                        votes[3] += 0.114833f;
                        votes[4] += 0.133971f;
                        votes[5] += 0.157895f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_8(const float features[8], float votes[6]) {
    if (features[4] <= 0.960270f) {
        if (features[1] <= -0.225485f) {
            if (features[5] <= 0.039676f) {
                if (features[5] <= -0.169043f) {
                    if (features[0] <= -5.886388f) {
                        votes[0] += 0.087838f;
                        votes[1] += 0.114865f;
                        votes[2] += 0.114865f;
                        votes[3] += 0.108108f;
                        votes[4] += 0.466216f;
                        votes[5] += 0.108108f;
                    } else {
                        votes[0] += 0.006202f;
                        votes[3] += 0.010078f;
                        votes[4] += 0.977907f;
                        votes[5] += 0.005814f;
                    }
                } else {
                    if (features[3] <= -0.419795f) {
                        votes[0] += 0.233918f;
                        votes[3] += 0.467836f;
                        votes[4] += 0.029240f;
                        votes[5] += 0.269006f;
                    } else {
                        votes[0] += 0.045267f;
                        votes[3] += 0.067490f;
                        votes[4] += 0.822222f;
                        votes[5] += 0.065021f;
                    }
                }
            } else {
                if (features[4] <= 0.719482f) {
                    if (features[4] <= -0.699988f) {
                        votes[0] += 0.303638f;
                        votes[3] += 0.312660f;
                        votes[4] += 0.079305f;
                        votes[5] += 0.304397f;
                    } else {
                        votes[0] += 0.099502f;
                        votes[3] += 0.146766f;
                        votes[4] += 0.597015f;
                        votes[5] += 0.156716f;
                    }
                } else {
                    votes[3] += 1.000000f;
                }
            }
        } else {
            if (features[1] <= -0.000493f) {
                if (features[3] <= -0.077490f) {
                    if (features[2] <= -0.150973f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[1] += 1.000000f;
                    }
                } else {
                    if (features[5] <= 0.293811f) {
                        votes[0] += 0.148936f;
                        votes[1] += 0.276596f;
                        votes[2] += 0.063830f;
                        votes[3] += 0.170213f;
                        votes[4] += 0.234043f;
                        votes[5] += 0.106383f;
                    } else {
                        votes[1] += 0.045455f;
                        votes[3] += 0.954545f;
                    }
                }
            } else {
                if (features[0] <= 0.223699f) {
                    if (features[3] <= 0.766525f) {
                        votes[0] += 0.222222f;
                        votes[1] += 0.095238f;
                        votes[2] += 0.015873f;
                        votes[3] += 0.281746f;
                        votes[4] += 0.210317f;
                        votes[5] += 0.174603f;
                    } else {
                        votes[0] += 0.164725f;
                        votes[1] += 0.171153f;
                        votes[2] += 0.177180f;
                        votes[3] += 0.162314f;
                        votes[4] += 0.162716f;
                        votes[5] += 0.161912f;
                    }
                } else {
                    if (features[1] <= 2.650557f) {
                        votes[0] += 0.005354f;
                        votes[1] += 0.005917f;
                        votes[2] += 0.970414f;
                        votes[3] += 0.005072f;
                        votes[4] += 0.005917f;
                        votes[5] += 0.007326f;
                    } else {
                        votes[0] += 0.200000f;
                        votes[1] += 0.142105f;
                        votes[2] += 0.200000f;
                        votes[3] += 0.194737f;
                        votes[4] += 0.110526f;
                        votes[5] += 0.152632f;
                    }
                }
            }
        }
    } else {
        if (features[3] <= 0.574433f) {
            if (features[7] <= 0.431272f) {
                if (features[5] <= -0.173313f) {
                    if (features[1] <= -0.211075f) {
                        votes[0] += 0.009056f;
                        votes[3] += 0.005175f;
                        votes[4] += 0.983182f;
                        votes[5] += 0.002587f;
                    } else {
                        votes[1] += 1.000000f;
                    }
                } else {
                    if (features[7] <= 0.349944f) {
                        votes[0] += 0.266941f;
                        votes[1] += 0.350385f;
                        votes[3] += 0.111214f;
                        votes[4] += 0.015413f;
                        votes[5] += 0.256046f;
                    } else {
                        votes[0] += 0.328216f;
                        votes[1] += 0.060389f;
                        votes[3] += 0.001365f;
                        votes[4] += 0.297509f;
                        votes[5] += 0.312521f;
                    }
                }
            } else {
                if (features[4] <= 1.328638f) {
                    if (features[0] <= 0.327434f) {
                        votes[0] += 0.001149f;
                        votes[3] += 0.996552f;
                        votes[5] += 0.002299f;
                    } else {
                        votes[1] += 0.428571f;
                        votes[4] += 0.285714f;
                        votes[5] += 0.285714f;
                    }
                } else {
                    if (features[3] <= 0.196703f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[1] += 0.600000f;
                        votes[3] += 0.400000f;
                    }
                }
            }
        } else {
            if (features[2] <= 0.062091f) {
                if (features[2] <= -0.204683f) {
                    if (features[7] <= -0.047935f) {
                        votes[0] += 0.133333f;
                        votes[1] += 0.800000f;
                        votes[2] += 0.066667f;
                    } else {
                        votes[0] += 0.040000f;
                        votes[1] += 0.200000f;
                        votes[2] += 0.180000f;
                        votes[3] += 0.540000f;
                        votes[5] += 0.040000f;
                    }
                } else {
                    if (features[2] <= 0.060209f) {
                        votes[0] += 0.241935f;
                        votes[1] += 0.169355f;
                        votes[2] += 0.233871f;
                        votes[3] += 0.088710f;
                        votes[4] += 0.016129f;
                        votes[5] += 0.250000f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                }
            } else {
                if (features[4] <= 1.314161f) {
                    if (features[4] <= 0.986886f) {
                        votes[0] += 0.266667f;
                        votes[1] += 0.333333f;
                        votes[2] += 0.133333f;
                        votes[5] += 0.266667f;
                    } else {
                        votes[0] += 0.003991f;
                        votes[1] += 0.005653f;
                        votes[2] += 0.981709f;
                        votes[3] += 0.002660f;
                        votes[4] += 0.000998f;
                        votes[5] += 0.004988f;
                    }
                } else {
                    if (features[5] <= -0.256793f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.215054f;
                        votes[1] += 0.268817f;
                        votes[2] += 0.172043f;
                        votes[3] += 0.150538f;
                        votes[4] += 0.032258f;
                        votes[5] += 0.161290f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_9(const float features[8], float votes[6]) {
    if (features[5] <= -0.116263f) {
        if (features[1] <= -0.219485f) {
            if (features[5] <= -0.169043f) {
                if (features[7] <= -6.491477f) {
                    if (features[3] <= -1.295565f) {
                        votes[0] += 0.058824f;
                        votes[1] += 0.105882f;
                        votes[2] += 0.047059f;
                        votes[3] += 0.082353f;
                        votes[4] += 0.635294f;
                        votes[5] += 0.070588f;
                    } else {
                        votes[0] += 0.106383f;
                        votes[1] += 0.212766f;
                        votes[2] += 0.148936f;
                        votes[3] += 0.106383f;
                        votes[4] += 0.191489f;
                        votes[5] += 0.234043f;
                    }
                } else {
                    if (features[7] <= 0.058665f) {
                        votes[0] += 0.003661f;
                        votes[4] += 0.989628f;
                        votes[5] += 0.006711f;
                    } else {
                        votes[0] += 0.036585f;
                        votes[3] += 0.201220f;
                        votes[4] += 0.707317f;
                        votes[5] += 0.054878f;
                    }
                }
            } else {
                if (features[7] <= 0.015089f) {
                    if (features[3] <= -0.430413f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.463320f;
                        votes[3] += 0.084942f;
                        votes[4] += 0.030888f;
                        votes[5] += 0.420849f;
                    }
                } else {
                    if (features[3] <= -0.330290f) {
                        votes[0] += 0.001637f;
                        votes[3] += 0.008183f;
                        votes[4] += 0.983633f;
                        votes[5] += 0.006547f;
                    } else {
                        votes[0] += 0.454545f;
                        votes[3] += 0.181818f;
                        votes[4] += 0.181818f;
                        votes[5] += 0.181818f;
                    }
                }
            }
        } else {
            if (features[0] <= 0.051901f) {
                if (features[2] <= -0.867818f) {
                    if (features[7] <= -6.324191f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.158223f;
                        votes[1] += 0.168189f;
                        votes[2] += 0.178571f;
                        votes[3] += 0.160299f;
                        votes[4] += 0.168605f;
                        votes[5] += 0.166113f;
                    }
                } else {
                    if (features[1] <= 1.018911f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.196429f;
                        votes[1] += 0.151786f;
                        votes[2] += 0.116071f;
                        votes[3] += 0.071429f;
                        votes[4] += 0.232143f;
                        votes[5] += 0.232143f;
                    }
                }
            } else {
                if (features[3] <= -0.069835f) {
                    votes[1] += 1.000000f;
                } else {
                    if (features[1] <= 1.245634f) {
                        votes[0] += 0.004622f;
                        votes[2] += 0.963020f;
                        votes[4] += 0.020031f;
                        votes[5] += 0.012327f;
                    } else {
                        votes[0] += 0.111111f;
                        votes[1] += 0.092593f;
                        votes[2] += 0.129630f;
                        votes[3] += 0.203704f;
                        votes[4] += 0.425926f;
                        votes[5] += 0.037037f;
                    }
                }
            }
        }
    } else {
        if (features[1] <= -0.226738f) {
            if (features[5] <= 0.142259f) {
                if (features[3] <= -0.387766f) {
                    if (features[7] <= 0.052654f) {
                        votes[0] += 0.425884f;
                        votes[3] += 0.070730f;
                        votes[4] += 0.068096f;
                        votes[5] += 0.435290f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                } else {
                    if (features[0] <= 0.300259f) {
                        votes[0] += 0.116505f;
                        votes[3] += 0.189320f;
                        votes[4] += 0.577670f;
                        votes[5] += 0.116505f;
                    } else {
                        votes[0] += 0.007864f;
                        votes[3] += 0.006655f;
                        votes[4] += 0.975197f;
                        votes[5] += 0.010284f;
                    }
                }
            } else {
                if (features[6] <= 0.978135f) {
                    if (features[7] <= 0.338120f) {
                        votes[0] += 0.348493f;
                        votes[3] += 0.267199f;
                        votes[4] += 0.031049f;
                        votes[5] += 0.353259f;
                    } else {
                        votes[0] += 0.003373f;
                        votes[3] += 0.991006f;
                        votes[4] += 0.000562f;
                        votes[5] += 0.005059f;
                    }
                } else {
                    if (features[4] <= 1.094232f) {
                        votes[0] += 0.488660f;
                        votes[3] += 0.002062f;
                        votes[5] += 0.509278f;
                    } else {
                        votes[0] += 0.328492f;
                        votes[1] += 0.000372f;
                        votes[3] += 0.312477f;
                        votes[4] += 0.021415f;
                        votes[5] += 0.337244f;
                    }
                }
            }
        } else {
            if (features[3] <= 0.597203f) {
                if (features[7] <= 0.363472f) {
                    if (features[2] <= -0.048218f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.004288f;
                        votes[1] += 0.990600f;
                        votes[3] += 0.001319f;
                        votes[5] += 0.003793f;
                    }
                } else {
                    if (features[5] <= 0.095903f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.125000f;
                        votes[1] += 0.534091f;
                        votes[3] += 0.102273f;
                        votes[4] += 0.056818f;
                        votes[5] += 0.181818f;
                    }
                }
            } else {
                if (features[3] <= 0.922156f) {
                    if (features[7] <= 0.510055f) {
                        votes[0] += 0.001612f;
                        votes[1] += 0.001791f;
                        votes[2] += 0.994268f;
                        votes[3] += 0.001254f;
                        votes[5] += 0.001075f;
                    } else {
                        votes[0] += 0.270270f;
                        votes[1] += 0.162162f;
                        votes[2] += 0.135135f;
                        votes[3] += 0.108108f;
                        votes[4] += 0.135135f;
                        votes[5] += 0.189189f;
                    }
                } else {
                    if (features[6] <= 0.838615f) {
                        votes[0] += 0.138350f;
                        votes[1] += 0.172330f;
                        votes[2] += 0.196602f;
                        votes[3] += 0.233010f;
                        votes[4] += 0.111650f;
                        votes[5] += 0.148058f;
                    } else {
                        votes[0] += 0.192982f;
                        votes[1] += 0.175439f;
                        votes[2] += 0.336257f;
                        votes[3] += 0.096491f;
                        votes[4] += 0.020468f;
                        votes[5] += 0.178363f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_10(const float features[8], float votes[6]) {
    if (features[1] <= -0.225485f) {
        if (features[7] <= 0.365449f) {
            if (features[3] <= -0.339649f) {
                if (features[3] <= -0.349885f) {
                    if (features[7] <= 0.052116f) {
                        votes[0] += 0.293512f;
                        votes[1] += 0.001429f;
                        votes[2] += 0.002001f;
                        votes[3] += 0.049062f;
                        votes[4] += 0.356483f;
                        votes[5] += 0.297514f;
                    } else {
                        votes[0] += 0.005183f;
                        votes[3] += 0.987042f;
                        votes[4] += 0.002592f;
                        votes[5] += 0.005183f;
                    }
                } else {
                    if (features[5] <= 0.156666f) {
                        votes[0] += 0.058824f;
                        votes[3] += 0.211765f;
                        votes[4] += 0.682353f;
                        votes[5] += 0.047059f;
                    } else {
                        votes[0] += 0.495453f;
                        votes[3] += 0.015215f;
                        votes[4] += 0.000175f;
                        votes[5] += 0.489157f;
                    }
                }
            } else {
                if (features[6] <= -0.871306f) {
                    if (features[2] <= 0.299700f) {
                        votes[0] += 0.117647f;
                        votes[3] += 0.058824f;
                        votes[4] += 0.823529f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[2] <= 0.301316f) {
                        votes[0] += 0.311573f;
                        votes[3] += 0.074184f;
                        votes[4] += 0.311573f;
                        votes[5] += 0.302671f;
                    } else {
                        votes[0] += 0.038560f;
                        votes[1] += 0.000857f;
                        votes[3] += 0.000857f;
                        votes[4] += 0.928877f;
                        votes[5] += 0.030848f;
                    }
                }
            }
        } else {
            if (features[7] <= 0.522899f) {
                if (features[3] <= -0.297650f) {
                    if (features[4] <= 1.040536f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.004790f;
                        votes[3] += 0.990419f;
                        votes[4] += 0.002395f;
                        votes[5] += 0.002395f;
                    }
                } else {
                    if (features[5] <= 0.059179f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.182796f;
                        votes[3] += 0.666667f;
                        votes[5] += 0.150538f;
                    }
                }
            } else {
                if (features[3] <= -0.314033f) {
                    votes[4] += 1.000000f;
                } else {
                    if (features[4] <= -0.839642f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.333333f;
                        votes[5] += 0.666667f;
                    }
                }
            }
        }
    } else {
        if (features[1] <= 0.487782f) {
            if (features[3] <= -0.101466f) {
                if (features[0] <= -0.208357f) {
                    votes[4] += 1.000000f;
                } else {
                    if (features[1] <= -0.195354f) {
                        votes[0] += 0.001863f;
                        votes[1] += 0.994410f;
                        votes[4] += 0.001242f;
                        votes[5] += 0.002484f;
                    } else {
                        votes[1] += 1.000000f;
                    }
                }
            } else {
                if (features[5] <= -0.145406f) {
                    if (features[2] <= -3.194929f) {
                        votes[0] += 0.212121f;
                        votes[1] += 0.212121f;
                        votes[2] += 0.181818f;
                        votes[3] += 0.121212f;
                        votes[4] += 0.030303f;
                        votes[5] += 0.242424f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[0] <= -0.116281f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.209424f;
                        votes[1] += 0.382199f;
                        votes[3] += 0.099476f;
                        votes[4] += 0.031414f;
                        votes[5] += 0.277487f;
                    }
                }
            }
        } else {
            if (features[0] <= 0.062044f) {
                if (features[1] <= 1.561007f) {
                    if (features[4] <= 0.840534f) {
                        votes[0] += 0.171053f;
                        votes[1] += 0.179511f;
                        votes[2] += 0.106673f;
                        votes[3] += 0.196898f;
                        votes[4] += 0.179041f;
                        votes[5] += 0.166823f;
                    } else {
                        votes[0] += 0.349206f;
                        votes[1] += 0.185185f;
                        votes[3] += 0.158730f;
                        votes[4] += 0.015873f;
                        votes[5] += 0.291005f;
                    }
                } else {
                    if (features[6] <= -1.544053f) {
                        votes[0] += 0.138298f;
                        votes[1] += 0.154255f;
                        votes[2] += 0.143617f;
                        votes[3] += 0.164894f;
                        votes[4] += 0.244681f;
                        votes[5] += 0.154255f;
                    } else {
                        votes[0] += 0.018450f;
                        votes[1] += 0.092251f;
                        votes[2] += 0.756458f;
                        votes[3] += 0.055351f;
                        votes[4] += 0.055351f;
                        votes[5] += 0.022140f;
                    }
                }
            } else {
                if (features[7] <= 0.375756f) {
                    if (features[2] <= 0.359171f) {
                        votes[0] += 0.002415f;
                        votes[1] += 0.003863f;
                        votes[2] += 0.988088f;
                        votes[3] += 0.001127f;
                        votes[4] += 0.002415f;
                        votes[5] += 0.002093f;
                    } else {
                        votes[0] += 0.100000f;
                        votes[1] += 0.350000f;
                        votes[2] += 0.200000f;
                        votes[3] += 0.250000f;
                        votes[5] += 0.100000f;
                    }
                } else {
                    if (features[5] <= -0.068640f) {
                        votes[0] += 0.052632f;
                        votes[3] += 0.052632f;
                        votes[4] += 0.868421f;
                        votes[5] += 0.026316f;
                    } else {
                        votes[0] += 0.209115f;
                        votes[1] += 0.123324f;
                        votes[2] += 0.230563f;
                        votes[3] += 0.139410f;
                        votes[4] += 0.093834f;
                        votes[5] += 0.203753f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_11(const float features[8], float votes[6]) {
    if (features[1] <= -0.225485f) {
        if (features[2] <= -0.353841f) {
            if (features[5] <= -3.655247f) {
                if (features[3] <= -1.295565f) {
                    if (features[1] <= -1.390372f) {
                        votes[0] += 0.118280f;
                        votes[1] += 0.043011f;
                        votes[2] += 0.064516f;
                        votes[3] += 0.075269f;
                        votes[4] += 0.612903f;
                        votes[5] += 0.086022f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[4] <= -2.693520f) {
                        votes[0] += 0.131148f;
                        votes[1] += 0.196721f;
                        votes[2] += 0.196721f;
                        votes[3] += 0.131148f;
                        votes[4] += 0.196721f;
                        votes[5] += 0.147541f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                }
            } else {
                votes[4] += 1.000000f;
            }
        } else {
            if (features[7] <= 0.366276f) {
                if (features[6] <= 0.975105f) {
                    if (features[5] <= -0.033934f) {
                        votes[0] += 0.028656f;
                        votes[3] += 0.087086f;
                        votes[4] += 0.861183f;
                        votes[5] += 0.023074f;
                    } else {
                        votes[0] += 0.316745f;
                        votes[3] += 0.253528f;
                        votes[4] += 0.108561f;
                        votes[5] += 0.321167f;
                    }
                } else {
                    if (features[3] <= -0.339835f) {
                        votes[0] += 0.377762f;
                        votes[3] += 0.121076f;
                        votes[4] += 0.127907f;
                        votes[5] += 0.373256f;
                    } else {
                        votes[0] += 0.155363f;
                        votes[1] += 0.000789f;
                        votes[3] += 0.003155f;
                        votes[4] += 0.688486f;
                        votes[5] += 0.152208f;
                    }
                }
            } else {
                if (features[2] <= 0.333857f) {
                    if (features[2] <= 0.246613f) {
                        votes[0] += 0.220588f;
                        votes[3] += 0.367647f;
                        votes[4] += 0.102941f;
                        votes[5] += 0.308824f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                } else {
                    if (features[7] <= 0.399880f) {
                        votes[0] += 0.307692f;
                        votes[4] += 0.461538f;
                        votes[5] += 0.230769f;
                    } else {
                        votes[0] += 0.250000f;
                        votes[5] += 0.750000f;
                    }
                }
            }
        }
    } else {
        if (features[1] <= 0.252469f) {
            if (features[0] <= -0.112375f) {
                if (features[2] <= -0.346871f) {
                    if (features[4] <= -2.664917f) {
                        votes[0] += 0.105263f;
                        votes[1] += 0.368421f;
                        votes[2] += 0.263158f;
                        votes[3] += 0.052632f;
                        votes[5] += 0.210526f;
                    } else {
                        votes[3] += 0.125000f;
                        votes[4] += 0.875000f;
                    }
                } else {
                    votes[3] += 1.000000f;
                }
            } else {
                if (features[3] <= -0.068404f) {
                    if (features[2] <= 0.346419f) {
                        votes[0] += 0.000487f;
                        votes[1] += 0.998701f;
                        votes[3] += 0.000487f;
                        votes[5] += 0.000325f;
                    } else {
                        votes[1] += 0.833333f;
                        votes[4] += 0.166667f;
                    }
                } else {
                    if (features[6] <= 0.088068f) {
                        votes[1] += 0.125000f;
                        votes[3] += 0.312500f;
                        votes[4] += 0.562500f;
                    } else {
                        votes[0] += 0.340426f;
                        votes[1] += 0.351064f;
                        votes[3] += 0.021277f;
                        votes[4] += 0.021277f;
                        votes[5] += 0.265957f;
                    }
                }
            }
        } else {
            if (features[6] <= -0.884202f) {
                if (features[3] <= 0.967611f) {
                    if (features[3] <= 0.289943f) {
                        votes[0] += 0.363636f;
                        votes[1] += 0.045455f;
                        votes[2] += 0.318182f;
                        votes[5] += 0.272727f;
                    } else {
                        votes[0] += 0.176471f;
                        votes[1] += 0.121569f;
                        votes[2] += 0.058824f;
                        votes[3] += 0.258824f;
                        votes[4] += 0.196078f;
                        votes[5] += 0.188235f;
                    }
                } else {
                    if (features[4] <= -1.014373f) {
                        votes[0] += 0.159681f;
                        votes[1] += 0.164671f;
                        votes[2] += 0.188623f;
                        votes[3] += 0.147705f;
                        votes[4] += 0.199601f;
                        votes[5] += 0.139721f;
                    } else {
                        votes[0] += 0.163743f;
                        votes[1] += 0.228070f;
                        votes[2] += 0.251462f;
                        votes[3] += 0.081871f;
                        votes[4] += 0.070175f;
                        votes[5] += 0.204678f;
                    }
                }
            } else {
                if (features[3] <= 0.791547f) {
                    if (features[0] <= -0.001375f) {
                        votes[0] += 0.223077f;
                        votes[1] += 0.061538f;
                        votes[3] += 0.276923f;
                        votes[4] += 0.161538f;
                        votes[5] += 0.276923f;
                    } else {
                        votes[0] += 0.001625f;
                        votes[1] += 0.003250f;
                        votes[2] += 0.993499f;
                        votes[3] += 0.000488f;
                        votes[4] += 0.000163f;
                        votes[5] += 0.000975f;
                    }
                } else {
                    if (features[6] <= 0.848559f) {
                        votes[0] += 0.135018f;
                        votes[1] += 0.158845f;
                        votes[2] += 0.215884f;
                        votes[3] += 0.166787f;
                        votes[4] += 0.187004f;
                        votes[5] += 0.136462f;
                    } else {
                        votes[0] += 0.153061f;
                        votes[1] += 0.200000f;
                        votes[2] += 0.332653f;
                        votes[3] += 0.095918f;
                        votes[4] += 0.038776f;
                        votes[5] += 0.179592f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_12(const float features[8], float votes[6]) {
    if (features[1] <= -0.225320f) {
        if (features[5] <= -0.022825f) {
            if (features[6] <= 0.762084f) {
                if (features[7] <= -3.314261f) {
                    if (features[2] <= -5.903965f) {
                        votes[0] += 0.028846f;
                        votes[1] += 0.067308f;
                        votes[2] += 0.076923f;
                        votes[3] += 0.048077f;
                        votes[4] += 0.692308f;
                        votes[5] += 0.086538f;
                    } else {
                        votes[0] += 0.152778f;
                        votes[1] += 0.194444f;
                        votes[2] += 0.152778f;
                        votes[3] += 0.250000f;
                        votes[4] += 0.097222f;
                        votes[5] += 0.152778f;
                    }
                } else {
                    if (features[1] <= -0.475530f) {
                        votes[0] += 0.004924f;
                        votes[3] += 0.000492f;
                        votes[4] += 0.991630f;
                        votes[5] += 0.002954f;
                    } else {
                        votes[0] += 0.039748f;
                        votes[3] += 0.032808f;
                        votes[4] += 0.886435f;
                        votes[5] += 0.041009f;
                    }
                }
            } else {
                if (features[2] <= 0.210123f) {
                    if (features[4] <= 0.972769f) {
                        votes[3] += 0.981928f;
                        votes[4] += 0.018072f;
                    } else {
                        votes[0] += 0.497159f;
                        votes[5] += 0.502841f;
                    }
                } else {
                    if (features[5] <= -0.089484f) {
                        votes[0] += 0.006475f;
                        votes[3] += 0.006475f;
                        votes[4] += 0.975948f;
                        votes[5] += 0.011101f;
                    } else {
                        votes[0] += 0.124378f;
                        votes[3] += 0.014925f;
                        votes[4] += 0.721393f;
                        votes[5] += 0.139303f;
                    }
                }
            }
        } else {
            if (features[2] <= -0.038606f) {
                if (features[6] <= 0.721420f) {
                    votes[4] += 1.000000f;
                } else {
                    votes[3] += 1.000000f;
                }
            } else {
                if (features[5] <= 0.142259f) {
                    if (features[1] <= -0.479900f) {
                        votes[0] += 0.413534f;
                        votes[3] += 0.132414f;
                        votes[4] += 0.055138f;
                        votes[5] += 0.398914f;
                    } else {
                        votes[0] += 0.068257f;
                        votes[3] += 0.042412f;
                        votes[4] += 0.826375f;
                        votes[5] += 0.062956f;
                    }
                } else {
                    if (features[0] <= 0.207659f) {
                        votes[0] += 0.502948f;
                        votes[4] += 0.001769f;
                        votes[5] += 0.495283f;
                    } else {
                        votes[0] += 0.307073f;
                        votes[1] += 0.000226f;
                        votes[3] += 0.354327f;
                        votes[4] += 0.024454f;
                        votes[5] += 0.313920f;
                    }
                }
            }
        }
    } else {
        if (features[0] <= 0.064575f) {
            if (features[2] <= -0.158040f) {
                if (features[1] <= 1.340399f) {
                    if (features[0] <= -0.350223f) {
                        votes[0] += 0.157193f;
                        votes[1] += 0.165893f;
                        votes[2] += 0.077726f;
                        votes[3] += 0.245360f;
                        votes[4] += 0.187355f;
                        votes[5] += 0.166473f;
                    } else {
                        votes[0] += 0.054545f;
                        votes[1] += 0.127273f;
                        votes[3] += 0.745455f;
                        votes[5] += 0.072727f;
                    }
                } else {
                    if (features[4] <= -1.325131f) {
                        votes[0] += 0.220238f;
                        votes[1] += 0.187500f;
                        votes[2] += 0.145833f;
                        votes[3] += 0.092262f;
                        votes[4] += 0.157738f;
                        votes[5] += 0.196429f;
                    } else {
                        votes[0] += 0.173984f;
                        votes[1] += 0.139837f;
                        votes[2] += 0.409756f;
                        votes[4] += 0.105691f;
                        votes[5] += 0.170732f;
                    }
                }
            } else {
                if (features[2] <= 0.024508f) {
                    if (features[1] <= 1.561007f) {
                        votes[0] += 0.250000f;
                        votes[1] += 0.296875f;
                        votes[4] += 0.062500f;
                        votes[5] += 0.390625f;
                    } else {
                        votes[0] += 0.116667f;
                        votes[1] += 0.066667f;
                        votes[2] += 0.516667f;
                        votes[3] += 0.033333f;
                        votes[4] += 0.116667f;
                        votes[5] += 0.150000f;
                    }
                } else {
                    if (features[6] <= 1.234255f) {
                        votes[1] += 0.923077f;
                        votes[4] += 0.076923f;
                    } else {
                        votes[0] += 0.333333f;
                        votes[1] += 0.083333f;
                        votes[3] += 0.416667f;
                        votes[5] += 0.166667f;
                    }
                }
            }
        } else {
            if (features[7] <= 0.330544f) {
                if (features[0] <= 0.304683f) {
                    if (features[3] <= 0.561109f) {
                        votes[0] += 0.004797f;
                        votes[1] += 0.989339f;
                        votes[4] += 0.000267f;
                        votes[5] += 0.005597f;
                    } else {
                        votes[0] += 0.005589f;
                        votes[1] += 0.005589f;
                        votes[2] += 0.976880f;
                        votes[3] += 0.003049f;
                        votes[4] += 0.002541f;
                        votes[5] += 0.006352f;
                    }
                } else {
                    if (features[6] <= 1.272347f) {
                        votes[0] += 0.001392f;
                        votes[1] += 0.052192f;
                        votes[2] += 0.945024f;
                        votes[3] += 0.001392f;
                    } else {
                        votes[1] += 0.501845f;
                        votes[2] += 0.487085f;
                        votes[4] += 0.011070f;
                    }
                }
            } else {
                if (features[3] <= 0.667496f) {
                    if (features[2] <= 0.298145f) {
                        votes[0] += 0.140351f;
                        votes[1] += 0.473684f;
                        votes[3] += 0.157895f;
                        votes[4] += 0.052632f;
                        votes[5] += 0.175439f;
                    } else {
                        votes[0] += 0.000885f;
                        votes[1] += 0.993805f;
                        votes[3] += 0.001327f;
                        votes[4] += 0.003540f;
                        votes[5] += 0.000442f;
                    }
                } else {
                    if (features[2] <= 0.349417f) {
                        votes[0] += 0.021576f;
                        votes[1] += 0.023452f;
                        votes[2] += 0.864916f;
                        votes[3] += 0.026266f;
                        votes[4] += 0.033771f;
                        votes[5] += 0.030019f;
                    } else {
                        votes[0] += 0.165217f;
                        votes[1] += 0.139130f;
                        votes[2] += 0.213043f;
                        votes[3] += 0.130435f;
                        votes[4] += 0.130435f;
                        votes[5] += 0.221739f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_13(const float features[8], float votes[6]) {
    if (features[0] <= 0.304686f) {
        if (features[7] <= 0.366297f) {
            if (features[2] <= -0.142071f) {
                if (features[0] <= -0.254238f) {
                    if (features[0] <= -0.588442f) {
                        votes[0] += 0.155797f;
                        votes[1] += 0.151087f;
                        votes[2] += 0.167391f;
                        votes[3] += 0.173188f;
                        votes[4] += 0.179348f;
                        votes[5] += 0.173188f;
                    } else {
                        votes[0] += 0.021790f;
                        votes[1] += 0.021790f;
                        votes[2] += 0.010895f;
                        votes[3] += 0.014786f;
                        votes[4] += 0.905058f;
                        votes[5] += 0.025681f;
                    }
                } else {
                    if (features[0] <= -0.208350f) {
                        votes[2] += 0.004753f;
                        votes[3] += 0.995247f;
                    } else {
                        votes[0] += 0.120000f;
                        votes[1] += 0.160000f;
                        votes[2] += 0.060000f;
                        votes[3] += 0.500000f;
                        votes[5] += 0.160000f;
                    }
                }
            } else {
                if (features[2] <= 0.300057f) {
                    if (features[2] <= 0.297523f) {
                        votes[0] += 0.221847f;
                        votes[1] += 0.230051f;
                        votes[2] += 0.243726f;
                        votes[3] += 0.025097f;
                        votes[4] += 0.059041f;
                        votes[5] += 0.220238f;
                    } else {
                        votes[0] += 0.054323f;
                        votes[1] += 0.069262f;
                        votes[2] += 0.757809f;
                        votes[3] += 0.020824f;
                        votes[4] += 0.037121f;
                        votes[5] += 0.060661f;
                    }
                } else {
                    if (features[1] <= -0.307258f) {
                        votes[0] += 0.302700f;
                        votes[3] += 0.148689f;
                        votes[4] += 0.251558f;
                        votes[5] += 0.297053f;
                    } else {
                        votes[1] += 0.751101f;
                        votes[2] += 0.248547f;
                        votes[4] += 0.000352f;
                    }
                }
            }
        } else {
            if (features[1] <= -0.163583f) {
                if (features[2] <= 0.246613f) {
                    if (features[5] <= 0.095175f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.267857f;
                        votes[3] += 0.428571f;
                        votes[4] += 0.017857f;
                        votes[5] += 0.285714f;
                    }
                } else {
                    votes[3] += 1.000000f;
                }
            } else {
                if (features[1] <= 0.872505f) {
                    if (features[5] <= -0.019882f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.144928f;
                        votes[1] += 0.492754f;
                        votes[3] += 0.144928f;
                        votes[4] += 0.086957f;
                        votes[5] += 0.130435f;
                    }
                } else {
                    if (features[5] <= 0.074260f) {
                        votes[0] += 0.025641f;
                        votes[1] += 0.102564f;
                        votes[3] += 0.025641f;
                        votes[4] += 0.794872f;
                        votes[5] += 0.051282f;
                    } else {
                        votes[0] += 0.165803f;
                        votes[1] += 0.170984f;
                        votes[2] += 0.274611f;
                        votes[3] += 0.139896f;
                        votes[4] += 0.025907f;
                        votes[5] += 0.222798f;
                    }
                }
            }
        }
    } else {
        if (features[7] <= 0.015978f) {
            votes[2] += 1.000000f;
        } else {
            if (features[1] <= 0.830332f) {
                if (features[3] <= -0.287400f) {
                    if (features[4] <= 1.098731f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.178674f;
                        votes[1] += 0.000961f;
                        votes[3] += 0.153698f;
                        votes[4] += 0.474544f;
                        votes[5] += 0.192123f;
                    }
                } else {
                    if (features[1] <= -0.231045f) {
                        votes[0] += 0.230769f;
                        votes[3] += 0.115385f;
                        votes[4] += 0.294872f;
                        votes[5] += 0.358974f;
                    } else {
                        votes[0] += 0.043702f;
                        votes[1] += 0.861183f;
                        votes[2] += 0.010283f;
                        votes[3] += 0.030848f;
                        votes[4] += 0.012853f;
                        votes[5] += 0.041131f;
                    }
                }
            } else {
                if (features[6] <= -0.812065f) {
                    if (features[6] <= -1.188875f) {
                        votes[0] += 0.194444f;
                        votes[1] += 0.166667f;
                        votes[4] += 0.333333f;
                        votes[5] += 0.305556f;
                    } else {
                        votes[0] += 0.117117f;
                        votes[1] += 0.135135f;
                        votes[2] += 0.279279f;
                        votes[3] += 0.207207f;
                        votes[4] += 0.153153f;
                        votes[5] += 0.108108f;
                    }
                } else {
                    if (features[1] <= 1.769315f) {
                        votes[0] += 0.001706f;
                        votes[2] += 0.994881f;
                        votes[5] += 0.003413f;
                    } else {
                        votes[0] += 0.156627f;
                        votes[1] += 0.144578f;
                        votes[2] += 0.120482f;
                        votes[3] += 0.180723f;
                        votes[4] += 0.228916f;
                        votes[5] += 0.168675f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_14(const float features[8], float votes[6]) {
    if (features[1] <= -0.225485f) {
        if (features[5] <= -0.022632f) {
            if (features[7] <= -0.013226f) {
                if (features[4] <= 0.765454f) {
                    if (features[2] <= -3.230425f) {
                        votes[0] += 0.106383f;
                        votes[1] += 0.127660f;
                        votes[2] += 0.106383f;
                        votes[3] += 0.154255f;
                        votes[4] += 0.452128f;
                        votes[5] += 0.053191f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[2] <= -0.060209f) {
                        votes[3] += 0.971429f;
                        votes[4] += 0.028571f;
                    } else {
                        votes[0] += 0.523952f;
                        votes[5] += 0.476048f;
                    }
                }
            } else {
                if (features[1] <= -0.480393f) {
                    if (features[7] <= 0.055276f) {
                        votes[0] += 0.002951f;
                        votes[4] += 0.994519f;
                        votes[5] += 0.002530f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                } else {
                    if (features[2] <= 0.301132f) {
                        votes[0] += 0.071579f;
                        votes[3] += 0.071579f;
                        votes[4] += 0.771579f;
                        votes[5] += 0.085263f;
                    } else {
                        votes[0] += 0.019175f;
                        votes[3] += 0.011505f;
                        votes[4] += 0.941515f;
                        votes[5] += 0.027804f;
                    }
                }
            }
        } else {
            if (features[2] <= -0.038606f) {
                if (features[6] <= 0.721420f) {
                    votes[4] += 1.000000f;
                } else {
                    votes[3] += 1.000000f;
                }
            } else {
                if (features[7] <= 0.367721f) {
                    if (features[7] <= 0.049954f) {
                        votes[0] += 0.496512f;
                        votes[1] += 0.000349f;
                        votes[4] += 0.007499f;
                        votes[5] += 0.495640f;
                    } else {
                        votes[0] += 0.289805f;
                        votes[3] += 0.246953f;
                        votes[4] += 0.159896f;
                        votes[5] += 0.303347f;
                    }
                } else {
                    if (features[2] <= 0.207944f) {
                        votes[0] += 0.533333f;
                        votes[4] += 0.133333f;
                        votes[5] += 0.333333f;
                    } else {
                        votes[0] += 0.003797f;
                        votes[3] += 0.992787f;
                        votes[4] += 0.001139f;
                        votes[5] += 0.002278f;
                    }
                }
            }
        }
    } else {
        if (features[3] <= -0.068404f) {
            if (features[7] <= 0.444876f) {
                if (features[4] <= -1.760565f) {
                    votes[3] += 1.000000f;
                } else {
                    if (features[7] <= -0.129440f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.000316f;
                        votes[1] += 0.998260f;
                        votes[3] += 0.000791f;
                        votes[5] += 0.000633f;
                    }
                }
            } else {
                votes[4] += 1.000000f;
            }
        } else {
            if (features[3] <= 0.785753f) {
                if (features[3] <= 0.599112f) {
                    if (features[2] <= -0.037715f) {
                        votes[0] += 0.120301f;
                        votes[1] += 0.097744f;
                        votes[2] += 0.060150f;
                        votes[3] += 0.278195f;
                        votes[4] += 0.255639f;
                        votes[5] += 0.187970f;
                    } else {
                        votes[0] += 0.179104f;
                        votes[1] += 0.350746f;
                        votes[3] += 0.126866f;
                        votes[4] += 0.089552f;
                        votes[5] += 0.253731f;
                    }
                } else {
                    if (features[7] <= -0.056135f) {
                        votes[0] += 0.191667f;
                        votes[1] += 0.141667f;
                        votes[2] += 0.066667f;
                        votes[3] += 0.208333f;
                        votes[4] += 0.191667f;
                        votes[5] += 0.200000f;
                    } else {
                        votes[0] += 0.002583f;
                        votes[1] += 0.001453f;
                        votes[2] += 0.988055f;
                        votes[3] += 0.002098f;
                        votes[4] += 0.003067f;
                        votes[5] += 0.002744f;
                    }
                }
            } else {
                if (features[1] <= 1.318799f) {
                    if (features[2] <= 0.053373f) {
                        votes[0] += 0.152381f;
                        votes[1] += 0.179487f;
                        votes[2] += 0.073993f;
                        votes[3] += 0.271795f;
                        votes[4] += 0.153846f;
                        votes[5] += 0.168498f;
                    } else {
                        votes[0] += 0.174419f;
                        votes[1] += 0.087209f;
                        votes[2] += 0.395349f;
                        votes[3] += 0.069767f;
                        votes[4] += 0.122093f;
                        votes[5] += 0.151163f;
                    }
                } else {
                    if (features[4] <= -1.031721f) {
                        votes[0] += 0.187380f;
                        votes[1] += 0.177820f;
                        votes[2] += 0.173996f;
                        votes[3] += 0.107075f;
                        votes[4] += 0.183556f;
                        votes[5] += 0.170172f;
                    } else {
                        votes[0] += 0.133090f;
                        votes[1] += 0.168642f;
                        votes[2] += 0.361896f;
                        votes[3] += 0.042844f;
                        votes[4] += 0.131267f;
                        votes[5] += 0.162261f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_15(const float features[8], float votes[6]) {
    if (features[5] <= -0.116263f) {
        if (features[1] <= -0.227080f) {
            if (features[0] <= 0.300279f) {
                if (features[2] <= -0.348646f) {
                    if (features[0] <= -5.886388f) {
                        votes[0] += 0.129730f;
                        votes[1] += 0.118919f;
                        votes[2] += 0.113514f;
                        votes[3] += 0.097297f;
                        votes[4] += 0.427027f;
                        votes[5] += 0.113514f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[5] <= -0.170329f) {
                        votes[0] += 0.084871f;
                        votes[3] += 0.059041f;
                        votes[4] += 0.793358f;
                        votes[5] += 0.062731f;
                    } else {
                        votes[0] += 0.341463f;
                        votes[3] += 0.243902f;
                        votes[4] += 0.104530f;
                        votes[5] += 0.310105f;
                    }
                }
            } else {
                if (features[0] <= 0.307314f) {
                    if (features[3] <= -0.422641f) {
                        votes[0] += 0.142857f;
                        votes[4] += 0.571429f;
                        votes[5] += 0.285714f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[0] <= 0.309390f) {
                        votes[0] += 0.230769f;
                        votes[3] += 0.076923f;
                        votes[4] += 0.653846f;
                        votes[5] += 0.038462f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                }
            }
        } else {
            if (features[2] <= 0.087035f) {
                if (features[2] <= -0.580509f) {
                    if (features[4] <= 0.722282f) {
                        votes[0] += 0.178266f;
                        votes[1] += 0.153439f;
                        votes[2] += 0.163207f;
                        votes[3] += 0.169719f;
                        votes[4] += 0.170940f;
                        votes[5] += 0.164428f;
                    } else {
                        votes[0] += 0.160000f;
                        votes[1] += 0.040000f;
                        votes[2] += 0.080000f;
                        votes[4] += 0.440000f;
                        votes[5] += 0.280000f;
                    }
                } else {
                    if (features[0] <= -0.445947f) {
                        votes[1] += 0.023810f;
                        votes[3] += 0.166667f;
                        votes[4] += 0.809524f;
                    } else {
                        votes[0] += 0.129032f;
                        votes[1] += 0.387097f;
                        votes[2] += 0.161290f;
                        votes[4] += 0.096774f;
                        votes[5] += 0.225806f;
                    }
                }
            } else {
                if (features[3] <= -0.069835f) {
                    votes[1] += 1.000000f;
                } else {
                    if (features[1] <= 1.245634f) {
                        votes[0] += 0.007937f;
                        votes[2] += 0.974603f;
                        votes[4] += 0.009524f;
                        votes[5] += 0.007937f;
                    } else {
                        votes[0] += 0.041667f;
                        votes[1] += 0.083333f;
                        votes[2] += 0.083333f;
                        votes[3] += 0.104167f;
                        votes[4] += 0.625000f;
                        votes[5] += 0.062500f;
                    }
                }
            }
        }
    } else {
        if (features[4] <= -0.883774f) {
            if (features[0] <= 0.299729f) {
                if (features[1] <= 1.134213f) {
                    if (features[6] <= -0.947268f) {
                        votes[0] += 0.666667f;
                        votes[5] += 0.333333f;
                    } else {
                        votes[0] += 0.062500f;
                        votes[1] += 0.406250f;
                        votes[3] += 0.187500f;
                        votes[4] += 0.218750f;
                        votes[5] += 0.125000f;
                    }
                } else {
                    if (features[3] <= 0.888773f) {
                        votes[2] += 1.000000f;
                    } else {
                        votes[0] += 0.033333f;
                        votes[1] += 0.266667f;
                        votes[2] += 0.400000f;
                        votes[3] += 0.166667f;
                        votes[5] += 0.133333f;
                    }
                }
            } else {
                if (features[3] <= -0.337705f) {
                    if (features[5] <= 0.365791f) {
                        votes[0] += 0.336898f;
                        votes[3] += 0.315508f;
                        votes[5] += 0.347594f;
                    } else {
                        votes[0] += 0.465160f;
                        votes[5] += 0.534840f;
                    }
                } else {
                    if (features[4] <= -0.913060f) {
                        votes[0] += 0.124088f;
                        votes[1] += 0.167883f;
                        votes[2] += 0.204380f;
                        votes[3] += 0.197080f;
                        votes[4] += 0.109489f;
                        votes[5] += 0.197080f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                }
            }
        } else {
            if (features[2] <= 0.304714f) {
                if (features[1] <= -0.226573f) {
                    if (features[5] <= 0.142259f) {
                        votes[0] += 0.240791f;
                        votes[3] += 0.129657f;
                        votes[4] += 0.371711f;
                        votes[5] += 0.257840f;
                    } else {
                        votes[0] += 0.309058f;
                        votes[3] += 0.359913f;
                        votes[4] += 0.021099f;
                        votes[5] += 0.309931f;
                    }
                } else {
                    if (features[1] <= 0.860815f) {
                        votes[0] += 0.006330f;
                        votes[1] += 0.981180f;
                        votes[3] += 0.006159f;
                        votes[4] += 0.001198f;
                        votes[5] += 0.005133f;
                    } else {
                        votes[0] += 0.024163f;
                        votes[1] += 0.020813f;
                        votes[2] += 0.910526f;
                        votes[3] += 0.022010f;
                        votes[4] += 0.001914f;
                        votes[5] += 0.020574f;
                    }
                }
            } else {
                if (features[7] <= 0.034381f) {
                    votes[2] += 1.000000f;
                } else {
                    if (features[5] <= 0.091835f) {
                        votes[0] += 0.090909f;
                        votes[1] += 0.278293f;
                        votes[2] += 0.092764f;
                        votes[3] += 0.029685f;
                        votes[4] += 0.437848f;
                        votes[5] += 0.070501f;
                    } else {
                        votes[0] += 0.139478f;
                        votes[1] += 0.121286f;
                        votes[2] += 0.474227f;
                        votes[3] += 0.127350f;
                        votes[4] += 0.023651f;
                        votes[5] += 0.114008f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_16(const float features[8], float votes[6]) {
    if (features[5] <= -0.116263f) {
        if (features[1] <= -0.219485f) {
            if (features[2] <= 0.300237f) {
                if (features[3] <= -0.388762f) {
                    if (features[5] <= -0.202044f) {
                        votes[0] += 0.031847f;
                        votes[1] += 0.016561f;
                        votes[2] += 0.028025f;
                        votes[3] += 0.040764f;
                        votes[4] += 0.848408f;
                        votes[5] += 0.034395f;
                    } else {
                        votes[0] += 0.389706f;
                        votes[3] += 0.268382f;
                        votes[4] += 0.029412f;
                        votes[5] += 0.312500f;
                    }
                } else {
                    if (features[5] <= -0.175193f) {
                        votes[0] += 0.022491f;
                        votes[3] += 0.015571f;
                        votes[4] += 0.953287f;
                        votes[5] += 0.008651f;
                    } else {
                        votes[0] += 0.304348f;
                        votes[3] += 0.173913f;
                        votes[4] += 0.369565f;
                        votes[5] += 0.152174f;
                    }
                }
            } else {
                if (features[2] <= 0.307313f) {
                    if (features[5] <= -0.168716f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.003289f;
                        votes[4] += 0.988487f;
                        votes[5] += 0.008224f;
                    }
                } else {
                    if (features[5] <= -0.687511f) {
                        votes[0] += 0.312500f;
                        votes[3] += 0.187500f;
                        votes[4] += 0.375000f;
                        votes[5] += 0.125000f;
                    } else {
                        votes[0] += 0.053097f;
                        votes[3] += 0.017699f;
                        votes[4] += 0.893805f;
                        votes[5] += 0.035398f;
                    }
                }
            }
        } else {
            if (features[3] <= 0.647864f) {
                if (features[1] <= 0.855340f) {
                    if (features[7] <= -0.128369f) {
                        votes[0] += 0.147727f;
                        votes[1] += 0.181818f;
                        votes[2] += 0.079545f;
                        votes[3] += 0.238636f;
                        votes[4] += 0.170455f;
                        votes[5] += 0.181818f;
                    } else {
                        votes[0] += 0.026490f;
                        votes[1] += 0.794702f;
                        votes[2] += 0.066225f;
                        votes[4] += 0.092715f;
                        votes[5] += 0.019868f;
                    }
                } else {
                    if (features[1] <= 1.073846f) {
                        votes[0] += 0.009539f;
                        votes[2] += 0.945946f;
                        votes[3] += 0.034976f;
                        votes[4] += 0.001590f;
                        votes[5] += 0.007949f;
                    } else {
                        votes[0] += 0.125000f;
                        votes[1] += 0.187500f;
                        votes[3] += 0.062500f;
                        votes[4] += 0.437500f;
                        votes[5] += 0.187500f;
                    }
                }
            } else {
                if (features[6] <= 1.108952f) {
                    if (features[0] <= 0.240064f) {
                        votes[0] += 0.170241f;
                        votes[1] += 0.170241f;
                        votes[2] += 0.180744f;
                        votes[3] += 0.159300f;
                        votes[4] += 0.168490f;
                        votes[5] += 0.150985f;
                    } else {
                        votes[0] += 0.029851f;
                        votes[1] += 0.029851f;
                        votes[2] += 0.716418f;
                        votes[3] += 0.059701f;
                        votes[4] += 0.134328f;
                        votes[5] += 0.029851f;
                    }
                } else {
                    if (features[0] <= 0.147720f) {
                        votes[0] += 0.190476f;
                        votes[1] += 0.095238f;
                        votes[2] += 0.285714f;
                        votes[4] += 0.047619f;
                        votes[5] += 0.380952f;
                    } else {
                        votes[2] += 0.740741f;
                        votes[4] += 0.259259f;
                    }
                }
            }
        }
    } else {
        if (features[3] <= -0.294107f) {
            if (features[3] <= -0.339576f) {
                if (features[3] <= -0.349884f) {
                    if (features[7] <= 0.053045f) {
                        votes[0] += 0.444261f;
                        votes[3] += 0.064879f;
                        votes[4] += 0.048097f;
                        votes[5] += 0.442763f;
                    } else {
                        votes[0] += 0.004155f;
                        votes[3] += 0.987188f;
                        votes[4] += 0.001385f;
                        votes[5] += 0.007271f;
                    }
                } else {
                    if (features[5] <= 0.156666f) {
                        votes[0] += 0.050000f;
                        votes[3] += 0.316667f;
                        votes[4] += 0.616667f;
                        votes[5] += 0.016667f;
                    } else {
                        votes[0] += 0.478268f;
                        votes[3] += 0.015011f;
                        votes[4] += 0.000873f;
                        votes[5] += 0.505847f;
                    }
                }
            } else {
                if (features[7] <= 0.368921f) {
                    if (features[7] <= 0.330004f) {
                        votes[0] += 0.364273f;
                        votes[3] += 0.054291f;
                        votes[4] += 0.162872f;
                        votes[5] += 0.418564f;
                    } else {
                        votes[0] += 0.039942f;
                        votes[4] += 0.922065f;
                        votes[5] += 0.037993f;
                    }
                } else {
                    if (features[3] <= -0.316083f) {
                        votes[0] += 0.000792f;
                        votes[3] += 0.997229f;
                        votes[5] += 0.001979f;
                    } else {
                        votes[0] += 0.060606f;
                        votes[3] += 0.787879f;
                        votes[4] += 0.090909f;
                        votes[5] += 0.060606f;
                    }
                }
            }
        } else {
            if (features[3] <= 0.597203f) {
                if (features[7] <= 0.362996f) {
                    if (features[6] <= -0.967065f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.017016f;
                        votes[1] += 0.954152f;
                        votes[3] += 0.008508f;
                        votes[4] += 0.005987f;
                        votes[5] += 0.014337f;
                    }
                } else {
                    if (features[2] <= 0.324552f) {
                        votes[0] += 0.133333f;
                        votes[1] += 0.166667f;
                        votes[3] += 0.441667f;
                        votes[4] += 0.083333f;
                        votes[5] += 0.175000f;
                    } else {
                        votes[0] += 0.407407f;
                        votes[1] += 0.259259f;
                        votes[3] += 0.018519f;
                        votes[4] += 0.129630f;
                        votes[5] += 0.185185f;
                    }
                }
            } else {
                if (features[1] <= 1.306253f) {
                    if (features[3] <= 0.796469f) {
                        votes[0] += 0.002123f;
                        votes[1] += 0.001592f;
                        votes[2] += 0.992570f;
                        votes[3] += 0.001238f;
                        votes[4] += 0.000354f;
                        votes[5] += 0.002123f;
                    } else {
                        votes[0] += 0.168000f;
                        votes[1] += 0.152000f;
                        votes[2] += 0.180000f;
                        votes[3] += 0.332000f;
                        votes[4] += 0.024000f;
                        votes[5] += 0.144000f;
                    }
                } else {
                    if (features[7] <= 0.467462f) {
                        votes[0] += 0.178248f;
                        votes[1] += 0.226586f;
                        votes[2] += 0.383686f;
                        votes[3] += 0.045317f;
                        votes[4] += 0.009063f;
                        votes[5] += 0.157100f;
                    } else {
                        votes[0] += 0.130137f;
                        votes[1] += 0.147260f;
                        votes[2] += 0.226027f;
                        votes[3] += 0.178082f;
                        votes[4] += 0.123288f;
                        votes[5] += 0.195205f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_17(const float features[8], float votes[6]) {
    if (features[7] <= 0.378691f) {
        if (features[1] <= -0.225485f) {
            if (features[3] <= -0.339576f) {
                if (features[1] <= -0.471296f) {
                    if (features[5] <= -0.062501f) {
                        votes[0] += 0.023077f;
                        votes[1] += 0.006689f;
                        votes[2] += 0.006689f;
                        votes[3] += 0.007358f;
                        votes[4] += 0.932441f;
                        votes[5] += 0.023746f;
                    } else {
                        votes[0] += 0.354782f;
                        votes[3] += 0.295158f;
                        votes[4] += 0.003184f;
                        votes[5] += 0.346876f;
                    }
                } else {
                    if (features[0] <= -0.074806f) {
                        votes[3] += 0.535461f;
                        votes[4] += 0.464539f;
                    } else {
                        votes[0] += 0.484258f;
                        votes[3] += 0.029049f;
                        votes[4] += 0.035310f;
                        votes[5] += 0.451383f;
                    }
                }
            } else {
                if (features[0] <= 0.299679f) {
                    if (features[5] <= 0.137021f) {
                        votes[0] += 0.136240f;
                        votes[3] += 0.054496f;
                        votes[4] += 0.673025f;
                        votes[5] += 0.136240f;
                    } else {
                        votes[0] += 0.446254f;
                        votes[3] += 0.136808f;
                        votes[4] += 0.053746f;
                        votes[5] += 0.363192f;
                    }
                } else {
                    if (features[0] <= 0.306225f) {
                        votes[0] += 0.010961f;
                        votes[1] += 0.001686f;
                        votes[4] += 0.978078f;
                        votes[5] += 0.009275f;
                    } else {
                        votes[0] += 0.320144f;
                        votes[3] += 0.007194f;
                        votes[4] += 0.341727f;
                        votes[5] += 0.330935f;
                    }
                }
            }
        } else {
            if (features[0] <= 0.007063f) {
                if (features[5] <= -0.170371f) {
                    if (features[1] <= 1.516280f) {
                        votes[0] += 0.173591f;
                        votes[1] += 0.180020f;
                        votes[2] += 0.103858f;
                        votes[3] += 0.166667f;
                        votes[4] += 0.201286f;
                        votes[5] += 0.174580f;
                    } else {
                        votes[0] += 0.094972f;
                        votes[1] += 0.114525f;
                        votes[2] += 0.480447f;
                        votes[3] += 0.114525f;
                        votes[4] += 0.131285f;
                        votes[5] += 0.064246f;
                    }
                } else {
                    if (features[0] <= -0.158032f) {
                        votes[0] += 0.124161f;
                        votes[1] += 0.147651f;
                        votes[2] += 0.204698f;
                        votes[3] += 0.352349f;
                        votes[4] += 0.010067f;
                        votes[5] += 0.161074f;
                    } else {
                        votes[0] += 0.250000f;
                        votes[1] += 0.350000f;
                        votes[2] += 0.175000f;
                        votes[5] += 0.225000f;
                    }
                }
            } else {
                if (features[6] <= -0.879577f) {
                    if (features[6] <= -0.880146f) {
                        votes[0] += 0.031579f;
                        votes[1] += 0.428070f;
                        votes[2] += 0.473684f;
                        votes[3] += 0.028070f;
                        votes[4] += 0.017544f;
                        votes[5] += 0.021053f;
                    } else {
                        votes[2] += 1.000000f;
                    }
                } else {
                    if (features[1] <= 0.650065f) {
                        votes[0] += 0.004560f;
                        votes[1] += 0.986808f;
                        votes[3] += 0.000163f;
                        votes[4] += 0.001954f;
                        votes[5] += 0.006515f;
                    } else {
                        votes[0] += 0.002814f;
                        votes[1] += 0.002814f;
                        votes[2] += 0.990233f;
                        votes[3] += 0.001159f;
                        votes[4] += 0.001490f;
                        votes[5] += 0.001490f;
                    }
                }
            }
        }
    } else {
        if (features[0] <= 0.318271f) {
            if (features[7] <= 0.455421f) {
                if (features[2] <= 0.266449f) {
                    if (features[6] <= 0.997965f) {
                        votes[0] += 0.046512f;
                        votes[1] += 0.116279f;
                        votes[2] += 0.209302f;
                        votes[3] += 0.488372f;
                        votes[4] += 0.023256f;
                        votes[5] += 0.116279f;
                    } else {
                        votes[0] += 0.288889f;
                        votes[1] += 0.288889f;
                        votes[2] += 0.266667f;
                        votes[4] += 0.044444f;
                        votes[5] += 0.111111f;
                    }
                } else {
                    if (features[2] <= 0.282182f) {
                        votes[0] += 0.266667f;
                        votes[3] += 0.666667f;
                        votes[5] += 0.066667f;
                    } else {
                        votes[0] += 0.000397f;
                        votes[1] += 0.001191f;
                        votes[3] += 0.997221f;
                        votes[4] += 0.001191f;
                    }
                }
            } else {
                if (features[2] <= 0.312027f) {
                    if (features[3] <= 13.317807f) {
                        votes[0] += 0.200000f;
                        votes[1] += 0.182222f;
                        votes[2] += 0.151111f;
                        votes[3] += 0.182222f;
                        votes[4] += 0.093333f;
                        votes[5] += 0.191111f;
                    } else {
                        votes[0] += 0.083333f;
                        votes[2] += 0.166667f;
                        votes[3] += 0.083333f;
                        votes[4] += 0.666667f;
                    }
                } else {
                    if (features[7] <= 0.735654f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[1] += 1.000000f;
                    }
                }
            }
        } else {
            if (features[2] <= 0.352120f) {
                if (features[4] <= 1.386473f) {
                    if (features[4] <= 1.363503f) {
                        votes[0] += 0.111111f;
                        votes[1] += 0.018519f;
                        votes[2] += 0.018519f;
                        votes[3] += 0.370370f;
                        votes[4] += 0.222222f;
                        votes[5] += 0.259259f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    votes[2] += 1.000000f;
                }
            } else {
                if (features[4] <= -1.212902f) {
                    if (features[5] <= 0.689544f) {
                        votes[1] += 0.181818f;
                        votes[4] += 0.818182f;
                    } else {
                        votes[0] += 0.600000f;
                        votes[5] += 0.400000f;
                    }
                } else {
                    if (features[5] <= 3.860923f) {
                        votes[0] += 0.157609f;
                        votes[1] += 0.228261f;
                        votes[2] += 0.201087f;
                        votes[3] += 0.125000f;
                        votes[4] += 0.076087f;
                        votes[5] += 0.211957f;
                    } else {
                        votes[0] += 0.480000f;
                        votes[3] += 0.160000f;
                        votes[4] += 0.160000f;
                        votes[5] += 0.200000f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_18(const float features[8], float votes[6]) {
    if (features[2] <= 0.304686f) {
        if (features[1] <= -0.225485f) {
            if (features[0] <= -0.353816f) {
                if (features[2] <= -3.230425f) {
                    if (features[5] <= -6.821496f) {
                        votes[0] += 0.050505f;
                        votes[1] += 0.080808f;
                        votes[2] += 0.090909f;
                        votes[3] += 0.040404f;
                        votes[4] += 0.686869f;
                        votes[5] += 0.050505f;
                    } else {
                        votes[0] += 0.200000f;
                        votes[1] += 0.142857f;
                        votes[2] += 0.171429f;
                        votes[3] += 0.171429f;
                        votes[4] += 0.185714f;
                        votes[5] += 0.128571f;
                    }
                } else {
                    votes[4] += 1.000000f;
                }
            } else {
                if (features[3] <= -0.334782f) {
                    if (features[4] <= 0.972769f) {
                        votes[0] += 0.269944f;
                        votes[3] += 0.229610f;
                        votes[4] += 0.232769f;
                        votes[5] += 0.267676f;
                    } else {
                        votes[0] += 0.359661f;
                        votes[3] += 0.104372f;
                        votes[4] += 0.180254f;
                        votes[5] += 0.355712f;
                    }
                } else {
                    if (features[2] <= 0.297911f) {
                        votes[0] += 0.241458f;
                        votes[3] += 0.257403f;
                        votes[4] += 0.231207f;
                        votes[5] += 0.269932f;
                    } else {
                        votes[3] += 0.997962f;
                        votes[4] += 0.002038f;
                    }
                }
            }
        } else {
            if (features[0] <= 0.300045f) {
                if (features[0] <= 0.297294f) {
                    if (features[1] <= 0.030156f) {
                        votes[0] += 0.017919f;
                        votes[1] += 0.929014f;
                        votes[2] += 0.002757f;
                        votes[3] += 0.017919f;
                        votes[4] += 0.017229f;
                        votes[5] += 0.015162f;
                    } else {
                        votes[0] += 0.113026f;
                        votes[1] += 0.121979f;
                        votes[2] += 0.430170f;
                        votes[3] += 0.109893f;
                        votes[4] += 0.104073f;
                        votes[5] += 0.120859f;
                    }
                } else {
                    if (features[5] <= 0.298072f) {
                        votes[1] += 0.057639f;
                        votes[2] += 0.942361f;
                    } else {
                        votes[1] += 0.818182f;
                        votes[2] += 0.181818f;
                    }
                }
            } else {
                if (features[5] <= -0.126676f) {
                    votes[2] += 1.000000f;
                } else {
                    if (features[3] <= 0.605427f) {
                        votes[1] += 1.000000f;
                    } else {
                        votes[2] += 1.000000f;
                    }
                }
            }
        }
    } else {
        if (features[3] <= 0.585575f) {
            if (features[0] <= 0.304972f) {
                if (features[1] <= -0.323834f) {
                    if (features[1] <= -0.519153f) {
                        votes[0] += 0.230769f;
                        votes[3] += 0.307692f;
                        votes[5] += 0.461538f;
                    } else {
                        votes[0] += 0.013550f;
                        votes[3] += 0.018970f;
                        votes[4] += 0.959350f;
                        votes[5] += 0.008130f;
                    }
                } else {
                    votes[1] += 1.000000f;
                }
            } else {
                if (features[6] <= 1.286109f) {
                    if (features[4] <= 1.283460f) {
                        votes[0] += 0.186522f;
                        votes[1] += 0.206980f;
                        votes[3] += 0.158845f;
                        votes[4] += 0.256318f;
                        votes[5] += 0.191336f;
                    } else {
                        votes[0] += 0.098039f;
                        votes[1] += 0.039216f;
                        votes[3] += 0.039216f;
                        votes[4] += 0.725490f;
                        votes[5] += 0.098039f;
                    }
                } else {
                    if (features[2] <= 0.305226f) {
                        votes[1] += 0.857143f;
                        votes[3] += 0.142857f;
                    } else {
                        votes[0] += 0.285714f;
                        votes[1] += 0.181818f;
                        votes[3] += 0.103896f;
                        votes[4] += 0.051948f;
                        votes[5] += 0.376623f;
                    }
                }
            }
        } else {
            if (features[5] <= 0.418063f) {
                if (features[7] <= 0.385800f) {
                    if (features[5] <= -0.642067f) {
                        votes[2] += 0.714286f;
                        votes[4] += 0.285714f;
                    } else {
                        votes[1] += 0.001761f;
                        votes[2] += 0.994718f;
                        votes[3] += 0.002934f;
                        votes[4] += 0.000587f;
                    }
                } else {
                    if (features[3] <= 5.113988f) {
                        votes[0] += 0.155172f;
                        votes[1] += 0.086207f;
                        votes[2] += 0.258621f;
                        votes[3] += 0.258621f;
                        votes[4] += 0.103448f;
                        votes[5] += 0.137931f;
                    } else {
                        votes[3] += 0.294118f;
                        votes[4] += 0.705882f;
                    }
                }
            } else {
                if (features[3] <= 3.178146f) {
                    if (features[1] <= 0.982192f) {
                        votes[0] += 0.368421f;
                        votes[1] += 0.105263f;
                        votes[3] += 0.157895f;
                        votes[4] += 0.105263f;
                        votes[5] += 0.263158f;
                    } else {
                        votes[0] += 0.014706f;
                        votes[1] += 0.102941f;
                        votes[2] += 0.779412f;
                        votes[4] += 0.029412f;
                        votes[5] += 0.073529f;
                    }
                } else {
                    if (features[3] <= 3.810608f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.250000f;
                        votes[1] += 0.267857f;
                        votes[2] += 0.098214f;
                        votes[3] += 0.142857f;
                        votes[4] += 0.080357f;
                        votes[5] += 0.160714f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_19(const float features[8], float votes[6]) {
    if (features[2] <= 0.304686f) {
        if (features[7] <= 0.366228f) {
            if (features[7] <= 0.330661f) {
                if (features[1] <= -0.226738f) {
                    if (features[5] <= -0.169043f) {
                        votes[0] += 0.009759f;
                        votes[1] += 0.008783f;
                        votes[2] += 0.007482f;
                        votes[3] += 0.013663f;
                        votes[4] += 0.947951f;
                        votes[5] += 0.012362f;
                    } else {
                        votes[0] += 0.319511f;
                        votes[3] += 0.298534f;
                        votes[4] += 0.062444f;
                        votes[5] += 0.319511f;
                    }
                } else {
                    if (features[0] <= 0.300045f) {
                        votes[0] += 0.075514f;
                        votes[1] += 0.281206f;
                        votes[2] += 0.442519f;
                        votes[3] += 0.068470f;
                        votes[4] += 0.058890f;
                        votes[5] += 0.073401f;
                    } else {
                        votes[1] += 0.744917f;
                        votes[2] += 0.254873f;
                        votes[4] += 0.000210f;
                    }
                }
            } else {
                if (features[1] <= -0.158993f) {
                    if (features[2] <= 0.304208f) {
                        votes[0] += 0.379826f;
                        votes[3] += 0.000890f;
                        votes[4] += 0.243373f;
                        votes[5] += 0.375912f;
                    } else {
                        votes[0] += 0.048913f;
                        votes[4] += 0.923913f;
                        votes[5] += 0.027174f;
                    }
                } else {
                    if (features[3] <= 0.678360f) {
                        votes[1] += 0.989664f;
                        votes[3] += 0.002584f;
                        votes[5] += 0.007752f;
                    } else {
                        votes[0] += 0.004854f;
                        votes[1] += 0.008091f;
                        votes[2] += 0.970874f;
                        votes[3] += 0.006472f;
                        votes[4] += 0.008091f;
                        votes[5] += 0.001618f;
                    }
                }
            }
        } else {
            if (features[3] <= -0.276710f) {
                if (features[2] <= 0.249742f) {
                    if (features[6] <= -0.881006f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.250000f;
                        votes[3] += 0.208333f;
                        votes[4] += 0.041667f;
                        votes[5] += 0.500000f;
                    }
                } else {
                    votes[3] += 1.000000f;
                }
            } else {
                if (features[7] <= 0.372951f) {
                    if (features[6] <= 1.065758f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[1] += 0.666667f;
                        votes[2] += 0.333333f;
                    }
                } else {
                    if (features[6] <= -1.044612f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.192926f;
                        votes[1] += 0.192926f;
                        votes[2] += 0.180064f;
                        votes[3] += 0.141479f;
                        votes[4] += 0.147910f;
                        votes[5] += 0.144695f;
                    }
                }
            }
        }
    } else {
        if (features[5] <= -0.105415f) {
            if (features[7] <= 0.034683f) {
                if (features[2] <= 0.305477f) {
                    votes[4] += 1.000000f;
                } else {
                    if (features[5] <= -0.299514f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[2] += 1.000000f;
                    }
                }
            } else {
                if (features[0] <= 0.305033f) {
                    votes[4] += 1.000000f;
                } else {
                    if (features[2] <= 0.313858f) {
                        votes[0] += 0.031847f;
                        votes[1] += 0.044586f;
                        votes[2] += 0.375796f;
                        votes[3] += 0.038217f;
                        votes[4] += 0.439490f;
                        votes[5] += 0.070064f;
                    } else {
                        votes[2] += 0.144578f;
                        votes[3] += 0.024096f;
                        votes[4] += 0.831325f;
                    }
                }
            }
        } else {
            if (features[6] <= -0.694981f) {
                if (features[7] <= 0.330604f) {
                    if (features[4] <= -1.061812f) {
                        votes[0] += 0.250000f;
                        votes[2] += 0.375000f;
                        votes[3] += 0.250000f;
                        votes[5] += 0.125000f;
                    } else {
                        votes[2] += 1.000000f;
                    }
                } else {
                    if (features[5] <= -0.032270f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.198529f;
                        votes[1] += 0.132353f;
                        votes[2] += 0.183824f;
                        votes[3] += 0.169118f;
                        votes[4] += 0.147059f;
                        votes[5] += 0.169118f;
                    }
                }
            } else {
                if (features[3] <= 0.617185f) {
                    if (features[3] <= -0.287400f) {
                        votes[0] += 0.271831f;
                        votes[3] += 0.246479f;
                        votes[4] += 0.235211f;
                        votes[5] += 0.246479f;
                    } else {
                        votes[0] += 0.055024f;
                        votes[1] += 0.842105f;
                        votes[3] += 0.026316f;
                        votes[4] += 0.026316f;
                        votes[5] += 0.050239f;
                    }
                } else {
                    if (features[7] <= 0.487804f) {
                        votes[2] += 1.000000f;
                    } else {
                        votes[0] += 0.213592f;
                        votes[1] += 0.203883f;
                        votes[2] += 0.213592f;
                        votes[3] += 0.106796f;
                        votes[4] += 0.058252f;
                        votes[5] += 0.203883f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_20(const float features[8], float votes[6]) {
    if (features[3] <= -0.294107f) {
        if (features[2] <= -0.353841f) {
            if (features[7] <= -3.420232f) {
                if (features[2] <= -5.903965f) {
                    if (features[7] <= -6.824097f) {
                        votes[0] += 0.030612f;
                        votes[1] += 0.040816f;
                        votes[2] += 0.091837f;
                        votes[3] += 0.132653f;
                        votes[4] += 0.653061f;
                        votes[5] += 0.051020f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[5] <= -6.798188f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.138462f;
                        votes[1] += 0.292308f;
                        votes[2] += 0.107692f;
                        votes[3] += 0.123077f;
                        votes[4] += 0.092308f;
                        votes[5] += 0.246154f;
                    }
                }
            } else {
                votes[4] += 1.000000f;
            }
        } else {
            if (features[7] <= 0.368921f) {
                if (features[3] <= -0.339668f) {
                    if (features[2] <= -0.074821f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.349094f;
                        votes[3] += 0.149814f;
                        votes[4] += 0.157075f;
                        votes[5] += 0.344017f;
                    }
                } else {
                    if (features[0] <= 0.300974f) {
                        votes[0] += 0.367713f;
                        votes[3] += 0.049327f;
                        votes[4] += 0.263079f;
                        votes[5] += 0.319880f;
                    } else {
                        votes[0] += 0.033975f;
                        votes[1] += 0.001637f;
                        votes[4] += 0.932869f;
                        votes[5] += 0.031519f;
                    }
                }
            } else {
                if (features[4] <= 1.309236f) {
                    if (features[7] <= 0.497397f) {
                        votes[0] += 0.000748f;
                        votes[3] += 0.998129f;
                        votes[4] += 0.001122f;
                    } else {
                        votes[0] += 0.222222f;
                        votes[4] += 0.666667f;
                        votes[5] += 0.111111f;
                    }
                } else {
                    votes[0] += 0.500000f;
                    votes[5] += 0.500000f;
                }
            }
        }
    } else {
        if (features[7] <= -0.031283f) {
            if (features[3] <= 2.169288f) {
                if (features[1] <= 1.261045f) {
                    if (features[1] <= 0.193401f) {
                        votes[0] += 0.125000f;
                        votes[1] += 0.025000f;
                        votes[2] += 0.025000f;
                        votes[3] += 0.050000f;
                        votes[4] += 0.725000f;
                        votes[5] += 0.050000f;
                    } else {
                        votes[0] += 0.151294f;
                        votes[1] += 0.134304f;
                        votes[2] += 0.077670f;
                        votes[3] += 0.299353f;
                        votes[4] += 0.173948f;
                        votes[5] += 0.163430f;
                    }
                } else {
                    if (features[3] <= 1.448515f) {
                        votes[0] += 0.105431f;
                        votes[1] += 0.182109f;
                        votes[2] += 0.466454f;
                        votes[3] += 0.028754f;
                        votes[4] += 0.115016f;
                        votes[5] += 0.102236f;
                    } else {
                        votes[0] += 0.206897f;
                        votes[1] += 0.222571f;
                        votes[2] += 0.100313f;
                        votes[3] += 0.061129f;
                        votes[4] += 0.217868f;
                        votes[5] += 0.191223f;
                    }
                }
            } else {
                if (features[1] <= 1.516258f) {
                    if (features[4] <= -0.928893f) {
                        votes[0] += 0.105263f;
                        votes[2] += 0.736842f;
                        votes[5] += 0.157895f;
                    } else {
                        votes[0] += 0.292683f;
                        votes[1] += 0.219512f;
                        votes[3] += 0.024390f;
                        votes[4] += 0.231707f;
                        votes[5] += 0.231707f;
                    }
                } else {
                    if (features[7] <= -3.805237f) {
                        votes[0] += 0.163934f;
                        votes[1] += 0.172131f;
                        votes[2] += 0.155738f;
                        votes[3] += 0.180328f;
                        votes[4] += 0.180328f;
                        votes[5] += 0.147541f;
                    } else {
                        votes[1] += 0.022989f;
                        votes[2] += 0.931034f;
                        votes[4] += 0.045977f;
                    }
                }
            }
        } else {
            if (features[3] <= 0.597203f) {
                if (features[0] <= -0.042699f) {
                    if (features[4] <= 0.735185f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[3] += 0.928571f;
                        votes[4] += 0.071429f;
                    }
                } else {
                    if (features[1] <= -0.238288f) {
                        votes[0] += 0.300813f;
                        votes[3] += 0.138211f;
                        votes[4] += 0.303523f;
                        votes[5] += 0.257453f;
                    } else {
                        votes[0] += 0.006700f;
                        votes[1] += 0.984263f;
                        votes[3] += 0.002805f;
                        votes[4] += 0.001091f;
                        votes[5] += 0.005142f;
                    }
                }
            } else {
                if (features[3] <= 0.918157f) {
                    if (features[1] <= 0.828475f) {
                        votes[0] += 0.196721f;
                        votes[1] += 0.131148f;
                        votes[2] += 0.131148f;
                        votes[3] += 0.114754f;
                        votes[4] += 0.180328f;
                        votes[5] += 0.245902f;
                    } else {
                        votes[0] += 0.000327f;
                        votes[1] += 0.001145f;
                        votes[2] += 0.996892f;
                        votes[3] += 0.000654f;
                        votes[4] += 0.000164f;
                        votes[5] += 0.000818f;
                    }
                } else {
                    if (features[2] <= -0.967876f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.166052f;
                        votes[1] += 0.159902f;
                        votes[2] += 0.233702f;
                        votes[3] += 0.164822f;
                        votes[4] += 0.100861f;
                        votes[5] += 0.174662f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_21(const float features[8], float votes[6]) {
    if (features[3] <= -0.294107f) {
        if (features[7] <= 0.368921f) {
            if (features[5] <= -0.023111f) {
                if (features[3] <= -0.422229f) {
                    if (features[4] <= 0.763443f) {
                        votes[0] += 0.034188f;
                        votes[1] += 0.028490f;
                        votes[2] += 0.038462f;
                        votes[3] += 0.019943f;
                        votes[4] += 0.840456f;
                        votes[5] += 0.038462f;
                    } else {
                        votes[0] += 0.287330f;
                        votes[3] += 0.357466f;
                        votes[4] += 0.006787f;
                        votes[5] += 0.348416f;
                    }
                } else {
                    if (features[5] <= -0.078389f) {
                        votes[0] += 0.020057f;
                        votes[1] += 0.001800f;
                        votes[3] += 0.008743f;
                        votes[4] += 0.945230f;
                        votes[5] += 0.024171f;
                    } else {
                        votes[0] += 0.110145f;
                        votes[3] += 0.075362f;
                        votes[4] += 0.669565f;
                        votes[5] += 0.144928f;
                    }
                }
            } else {
                if (features[3] <= -0.339576f) {
                    if (features[7] <= 0.285006f) {
                        votes[0] += 0.313276f;
                        votes[3] += 0.356163f;
                        votes[4] += 0.012639f;
                        votes[5] += 0.317922f;
                    } else {
                        votes[0] += 0.487848f;
                        votes[3] += 0.008393f;
                        votes[4] += 0.004022f;
                        votes[5] += 0.499738f;
                    }
                } else {
                    if (features[7] <= 0.330094f) {
                        votes[0] += 0.368836f;
                        votes[3] += 0.063116f;
                        votes[4] += 0.161736f;
                        votes[5] += 0.406312f;
                    } else {
                        votes[0] += 0.031863f;
                        votes[3] += 0.002451f;
                        votes[4] += 0.925858f;
                        votes[5] += 0.039828f;
                    }
                }
            }
        } else {
            if (features[0] <= 0.338110f) {
                if (features[3] <= -0.357126f) {
                    votes[4] += 1.000000f;
                } else {
                    if (features[7] <= 0.497397f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.428571f;
                        votes[4] += 0.285714f;
                        votes[5] += 0.285714f;
                    }
                }
            } else {
                if (features[5] <= 0.303689f) {
                    votes[4] += 1.000000f;
                } else {
                    if (features[7] <= 0.395173f) {
                        votes[0] += 0.666667f;
                        votes[5] += 0.333333f;
                    } else {
                        votes[0] += 0.285714f;
                        votes[5] += 0.714286f;
                    }
                }
            }
        }
    } else {
        if (features[6] <= -0.884947f) {
            if (features[1] <= 1.124443f) {
                if (features[5] <= -4.228777f) {
                    if (features[3] <= 0.787777f) {
                        votes[0] += 0.102564f;
                        votes[1] += 0.192308f;
                        votes[2] += 0.064103f;
                        votes[3] += 0.205128f;
                        votes[4] += 0.282051f;
                        votes[5] += 0.153846f;
                    } else {
                        votes[0] += 0.143590f;
                        votes[1] += 0.169231f;
                        votes[2] += 0.261538f;
                        votes[3] += 0.148718f;
                        votes[4] += 0.082051f;
                        votes[5] += 0.194872f;
                    }
                } else {
                    if (features[0] <= -3.903407f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.167702f;
                        votes[1] += 0.155280f;
                        votes[2] += 0.074534f;
                        votes[3] += 0.354037f;
                        votes[4] += 0.149068f;
                        votes[5] += 0.099379f;
                    }
                }
            } else {
                if (features[1] <= 1.136897f) {
                    if (features[0] <= -1.871597f) {
                        votes[3] += 0.052632f;
                        votes[4] += 0.947368f;
                    } else {
                        votes[0] += 1.000000f;
                    }
                } else {
                    if (features[0] <= -4.563709f) {
                        votes[0] += 0.172185f;
                        votes[1] += 0.211921f;
                        votes[2] += 0.149007f;
                        votes[3] += 0.135762f;
                        votes[4] += 0.165563f;
                        votes[5] += 0.165563f;
                    } else {
                        votes[0] += 0.167842f;
                        votes[1] += 0.146685f;
                        votes[2] += 0.244006f;
                        votes[3] += 0.086037f;
                        votes[4] += 0.181946f;
                        votes[5] += 0.173484f;
                    }
                }
            }
        } else {
            if (features[1] <= 0.525490f) {
                if (features[1] <= -0.226738f) {
                    if (features[6] <= 0.978135f) {
                        votes[0] += 0.174658f;
                        votes[3] += 0.232877f;
                        votes[4] += 0.421233f;
                        votes[5] += 0.171233f;
                    } else {
                        votes[0] += 0.453947f;
                        votes[1] += 0.006579f;
                        votes[3] += 0.072368f;
                        votes[4] += 0.125000f;
                        votes[5] += 0.342105f;
                    }
                } else {
                    if (features[2] <= -0.069592f) {
                        votes[3] += 0.489796f;
                        votes[4] += 0.510204f;
                    } else {
                        votes[0] += 0.004246f;
                        votes[1] += 0.984651f;
                        votes[3] += 0.002286f;
                        votes[4] += 0.001470f;
                        votes[5] += 0.007348f;
                    }
                }
            } else {
                if (features[2] <= 0.062091f) {
                    if (features[1] <= 1.487363f) {
                        votes[0] += 0.189105f;
                        votes[1] += 0.161089f;
                        votes[2] += 0.057588f;
                        votes[3] += 0.210895f;
                        votes[4] += 0.191440f;
                        votes[5] += 0.189883f;
                    } else {
                        votes[0] += 0.032727f;
                        votes[1] += 0.160000f;
                        votes[2] += 0.683636f;
                        votes[3] += 0.043636f;
                        votes[4] += 0.058182f;
                        votes[5] += 0.021818f;
                    }
                } else {
                    if (features[7] <= 0.518050f) {
                        votes[0] += 0.002168f;
                        votes[1] += 0.003561f;
                        votes[2] += 0.989316f;
                        votes[3] += 0.000774f;
                        votes[4] += 0.001858f;
                        votes[5] += 0.002323f;
                    } else {
                        votes[0] += 0.130435f;
                        votes[1] += 0.142857f;
                        votes[2] += 0.229814f;
                        votes[3] += 0.142857f;
                        votes[4] += 0.130435f;
                        votes[5] += 0.223602f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_22(const float features[8], float votes[6]) {
    if (features[7] <= 0.378691f) {
        if (features[7] <= 0.330661f) {
            if (features[0] <= 0.304738f) {
                if (features[3] <= -0.294496f) {
                    if (features[4] <= 0.972769f) {
                        votes[0] += 0.211582f;
                        votes[1] += 0.001655f;
                        votes[2] += 0.001363f;
                        votes[3] += 0.271338f;
                        votes[4] += 0.298589f;
                        votes[5] += 0.215474f;
                    } else {
                        votes[0] += 0.367583f;
                        votes[3] += 0.145580f;
                        votes[4] += 0.134185f;
                        votes[5] += 0.352652f;
                    }
                } else {
                    if (features[3] <= 0.128908f) {
                        votes[0] += 0.018374f;
                        votes[1] += 0.932071f;
                        votes[2] += 0.000186f;
                        votes[3] += 0.009465f;
                        votes[4] += 0.023014f;
                        votes[5] += 0.016889f;
                    } else {
                        votes[0] += 0.071793f;
                        votes[1] += 0.072377f;
                        votes[2] += 0.646432f;
                        votes[3] += 0.069313f;
                        votes[4] += 0.066102f;
                        votes[5] += 0.073982f;
                    }
                }
            } else {
                if (features[3] <= 0.459247f) {
                    if (features[1] <= -0.239105f) {
                        votes[0] += 0.182777f;
                        votes[3] += 0.165202f;
                        votes[4] += 0.465729f;
                        votes[5] += 0.186292f;
                    } else {
                        votes[0] += 0.014151f;
                        votes[1] += 0.981132f;
                        votes[5] += 0.004717f;
                    }
                } else {
                    if (features[1] <= 3.751432f) {
                        votes[2] += 0.998595f;
                        votes[3] += 0.000703f;
                        votes[4] += 0.000703f;
                    } else {
                        votes[1] += 0.571429f;
                        votes[4] += 0.142857f;
                        votes[5] += 0.285714f;
                    }
                }
            }
        } else {
            if (features[3] <= -0.340068f) {
                if (features[0] <= 0.300828f) {
                    if (features[5] <= 0.410126f) {
                        votes[5] += 1.000000f;
                    } else {
                        votes[0] += 0.200000f;
                        votes[5] += 0.800000f;
                    }
                } else {
                    if (features[6] <= -0.814364f) {
                        votes[0] += 0.499617f;
                        votes[5] += 0.500383f;
                    } else {
                        votes[0] += 0.472557f;
                        votes[4] += 0.000669f;
                        votes[5] += 0.526774f;
                    }
                }
            } else {
                if (features[4] <= 0.049774f) {
                    if (features[3] <= -0.322113f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.032258f;
                        votes[1] += 0.080645f;
                        votes[2] += 0.080645f;
                        votes[3] += 0.645161f;
                        votes[4] += 0.129032f;
                        votes[5] += 0.032258f;
                    }
                } else {
                    if (features[2] <= 0.300083f) {
                        votes[0] += 0.036082f;
                        votes[1] += 0.048969f;
                        votes[2] += 0.842784f;
                        votes[3] += 0.041237f;
                        votes[5] += 0.030928f;
                    } else {
                        votes[0] += 0.040963f;
                        votes[1] += 0.326858f;
                        votes[2] += 0.231841f;
                        votes[4] += 0.360220f;
                        votes[5] += 0.040118f;
                    }
                }
            }
        }
    } else {
        if (features[7] <= 0.451158f) {
            if (features[1] <= -0.222807f) {
                if (features[2] <= 0.260696f) {
                    if (features[7] <= 0.380704f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.466667f;
                        votes[3] += 0.100000f;
                        votes[4] += 0.066667f;
                        votes[5] += 0.366667f;
                    }
                } else {
                    if (features[4] <= 1.309236f) {
                        votes[3] += 0.999613f;
                        votes[5] += 0.000387f;
                    } else {
                        votes[0] += 0.333333f;
                        votes[5] += 0.666667f;
                    }
                }
            } else {
                if (features[5] <= -0.254182f) {
                    votes[4] += 1.000000f;
                } else {
                    if (features[1] <= 0.518871f) {
                        votes[1] += 0.923077f;
                        votes[4] += 0.076923f;
                    } else {
                        votes[0] += 0.166667f;
                        votes[1] += 0.138889f;
                        votes[2] += 0.291667f;
                        votes[3] += 0.166667f;
                        votes[5] += 0.236111f;
                    }
                }
            }
        } else {
            if (features[1] <= 1.211127f) {
                if (features[1] <= 0.822996f) {
                    if (features[5] <= 0.064654f) {
                        votes[1] += 0.071429f;
                        votes[4] += 0.928571f;
                    } else {
                        votes[0] += 0.175000f;
                        votes[1] += 0.191667f;
                        votes[3] += 0.325000f;
                        votes[4] += 0.075000f;
                        votes[5] += 0.233333f;
                    }
                } else {
                    if (features[5] <= 2.702952f) {
                        votes[0] += 0.500000f;
                        votes[4] += 0.029412f;
                        votes[5] += 0.470588f;
                    } else {
                        votes[1] += 1.000000f;
                    }
                }
            } else {
                if (features[5] <= 0.077917f) {
                    if (features[6] <= 1.497850f) {
                        votes[2] += 0.166667f;
                        votes[3] += 0.166667f;
                        votes[4] += 0.666667f;
                    } else {
                        votes[1] += 0.666667f;
                        votes[4] += 0.333333f;
                    }
                } else {
                    if (features[5] <= 4.936506f) {
                        votes[0] += 0.182482f;
                        votes[1] += 0.211679f;
                        votes[2] += 0.284672f;
                        votes[3] += 0.167883f;
                        votes[4] += 0.025547f;
                        votes[5] += 0.127737f;
                    } else {
                        votes[2] += 0.055556f;
                        votes[3] += 0.111111f;
                        votes[4] += 0.833333f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_23(const float features[8], float votes[6]) {
    if (features[1] <= -0.225485f) {
        if (features[7] <= 0.370648f) {
            if (features[7] <= 0.285006f) {
                if (features[7] <= 0.049553f) {
                    if (features[5] <= -0.146419f) {
                        votes[0] += 0.015757f;
                        votes[1] += 0.008723f;
                        votes[2] += 0.003376f;
                        votes[3] += 0.028981f;
                        votes[4] += 0.919809f;
                        votes[5] += 0.023354f;
                    } else {
                        votes[0] += 0.436775f;
                        votes[3] += 0.070314f;
                        votes[4] += 0.053415f;
                        votes[5] += 0.439496f;
                    }
                } else {
                    if (features[4] <= 1.306353f) {
                        votes[0] += 0.007629f;
                        votes[3] += 0.978703f;
                        votes[4] += 0.006357f;
                        votes[5] += 0.007311f;
                    } else {
                        votes[0] += 0.428571f;
                        votes[5] += 0.571429f;
                    }
                }
            } else {
                if (features[1] <= -0.433927f) {
                    if (features[5] <= 0.280696f) {
                        votes[0] += 0.001848f;
                        votes[4] += 0.995379f;
                        votes[5] += 0.002773f;
                    } else {
                        votes[0] += 0.507428f;
                        votes[4] += 0.001793f;
                        votes[5] += 0.490779f;
                    }
                } else {
                    if (features[5] <= 0.152002f) {
                        votes[0] += 0.013709f;
                        votes[3] += 0.006093f;
                        votes[4] += 0.964204f;
                        votes[5] += 0.015994f;
                    } else {
                        votes[0] += 0.418562f;
                        votes[3] += 0.033382f;
                        votes[4] += 0.118122f;
                        votes[5] += 0.429934f;
                    }
                }
            }
        } else {
            if (features[7] <= 0.522899f) {
                if (features[0] <= 0.247658f) {
                    if (features[0] <= 0.225323f) {
                        votes[0] += 0.347826f;
                        votes[3] += 0.391304f;
                        votes[5] += 0.260870f;
                    } else {
                        votes[0] += 0.590909f;
                        votes[4] += 0.090909f;
                        votes[5] += 0.318182f;
                    }
                } else {
                    if (features[5] <= 0.740376f) {
                        votes[0] += 0.001190f;
                        votes[3] += 0.997619f;
                        votes[4] += 0.000794f;
                        votes[5] += 0.000397f;
                    } else {
                        votes[0] += 0.027778f;
                        votes[3] += 0.944444f;
                        votes[5] += 0.027778f;
                    }
                }
            } else {
                if (features[5] <= 0.466096f) {
                    votes[4] += 1.000000f;
                } else {
                    if (features[4] <= -0.829010f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.600000f;
                        votes[5] += 0.400000f;
                    }
                }
            }
        }
    } else {
        if (features[3] <= 0.457679f) {
            if (features[2] <= -0.043838f) {
                if (features[6] <= 0.725628f) {
                    if (features[1] <= 0.122043f) {
                        votes[0] += 0.125000f;
                        votes[1] += 0.100000f;
                        votes[2] += 0.100000f;
                        votes[3] += 0.050000f;
                        votes[4] += 0.525000f;
                        votes[5] += 0.100000f;
                    } else {
                        votes[0] += 0.100000f;
                        votes[1] += 0.200000f;
                        votes[2] += 0.050000f;
                        votes[3] += 0.350000f;
                        votes[5] += 0.300000f;
                    }
                } else {
                    if (features[5] <= -0.269857f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                }
            } else {
                if (features[4] <= -0.969720f) {
                    votes[3] += 1.000000f;
                } else {
                    if (features[2] <= 0.432672f) {
                        votes[0] += 0.003777f;
                        votes[1] += 0.990244f;
                        votes[3] += 0.000315f;
                        votes[4] += 0.000315f;
                        votes[5] += 0.005350f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                }
            }
        } else {
            if (features[0] <= 0.062044f) {
                if (features[1] <= 1.419010f) {
                    if (features[5] <= -0.328257f) {
                        votes[0] += 0.166574f;
                        votes[1] += 0.173769f;
                        votes[2] += 0.087438f;
                        votes[3] += 0.199779f;
                        votes[4] += 0.195905f;
                        votes[5] += 0.176536f;
                    } else {
                        votes[0] += 0.278378f;
                        votes[1] += 0.162162f;
                        votes[3] += 0.256757f;
                        votes[4] += 0.054054f;
                        votes[5] += 0.248649f;
                    }
                } else {
                    if (features[7] <= -3.417889f) {
                        votes[0] += 0.165354f;
                        votes[1] += 0.165354f;
                        votes[2] += 0.141732f;
                        votes[3] += 0.129921f;
                        votes[4] += 0.192913f;
                        votes[5] += 0.204724f;
                    } else {
                        votes[0] += 0.052632f;
                        votes[1] += 0.170426f;
                        votes[2] += 0.621554f;
                        votes[3] += 0.030075f;
                        votes[4] += 0.060150f;
                        votes[5] += 0.065163f;
                    }
                }
            } else {
                if (features[3] <= 0.985071f) {
                    if (features[1] <= 0.827353f) {
                        votes[0] += 0.301370f;
                        votes[1] += 0.287671f;
                        votes[2] += 0.095890f;
                        votes[3] += 0.041096f;
                        votes[4] += 0.109589f;
                        votes[5] += 0.164384f;
                    } else {
                        votes[1] += 0.000161f;
                        votes[2] += 0.998714f;
                        votes[3] += 0.000321f;
                        votes[4] += 0.000643f;
                        votes[5] += 0.000161f;
                    }
                } else {
                    if (features[2] <= 0.280501f) {
                        votes[0] += 0.107955f;
                        votes[1] += 0.147727f;
                        votes[2] += 0.403409f;
                        votes[3] += 0.107955f;
                        votes[4] += 0.073864f;
                        votes[5] += 0.159091f;
                    } else {
                        votes[0] += 0.189286f;
                        votes[1] += 0.185714f;
                        votes[2] += 0.167857f;
                        votes[3] += 0.107143f;
                        votes[4] += 0.157143f;
                        votes[5] += 0.192857f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_24(const float features[8], float votes[6]) {
    if (features[3] <= -0.294107f) {
        if (features[6] <= 0.973283f) {
            if (features[3] <= -0.335092f) {
                if (features[5] <= -0.033934f) {
                    if (features[0] <= 0.301133f) {
                        votes[0] += 0.051692f;
                        votes[1] += 0.011077f;
                        votes[2] += 0.011692f;
                        votes[3] += 0.136000f;
                        votes[4] += 0.736615f;
                        votes[5] += 0.052923f;
                    } else {
                        votes[0] += 0.001989f;
                        votes[3] += 0.000995f;
                        votes[4] += 0.997016f;
                    }
                } else {
                    if (features[7] <= 0.317381f) {
                        votes[0] += 0.277983f;
                        votes[3] += 0.421079f;
                        votes[4] += 0.022453f;
                        votes[5] += 0.278485f;
                    } else {
                        votes[0] += 0.381051f;
                        votes[3] += 0.008488f;
                        votes[4] += 0.242028f;
                        votes[5] += 0.368433f;
                    }
                }
            } else {
                if (features[7] <= 0.337551f) {
                    if (features[5] <= 0.109581f) {
                        votes[0] += 0.062500f;
                        votes[3] += 0.027778f;
                        votes[4] += 0.791667f;
                        votes[5] += 0.118056f;
                    } else {
                        votes[0] += 0.408560f;
                        votes[3] += 0.093385f;
                        votes[4] += 0.093385f;
                        votes[5] += 0.404669f;
                    }
                } else {
                    if (features[5] <= 0.091805f) {
                        votes[3] += 0.888889f;
                        votes[4] += 0.111111f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                }
            }
        } else {
            if (features[4] <= 1.105770f) {
                if (features[0] <= 0.301665f) {
                    if (features[0] <= -0.039122f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.504615f;
                        votes[5] += 0.495385f;
                    }
                } else {
                    if (features[5] <= 0.594818f) {
                        votes[0] += 0.447257f;
                        votes[3] += 0.118143f;
                        votes[4] += 0.029536f;
                        votes[5] += 0.405063f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                }
            } else {
                if (features[3] <= -0.420679f) {
                    if (features[6] <= 1.106059f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.498408f;
                        votes[4] += 0.002123f;
                        votes[5] += 0.499469f;
                    }
                } else {
                    if (features[7] <= 0.424284f) {
                        votes[0] += 0.229785f;
                        votes[1] += 0.000853f;
                        votes[3] += 0.184340f;
                        votes[4] += 0.368893f;
                        votes[5] += 0.216130f;
                    } else {
                        votes[0] += 0.001379f;
                        votes[3] += 0.998621f;
                    }
                }
            }
        }
    } else {
        if (features[1] <= 0.525490f) {
            if (features[1] <= -0.225485f) {
                if (features[6] <= 0.978135f) {
                    if (features[0] <= 0.223273f) {
                        votes[3] += 0.469027f;
                        votes[4] += 0.530973f;
                    } else {
                        votes[0] += 0.303483f;
                        votes[3] += 0.129353f;
                        votes[4] += 0.298507f;
                        votes[5] += 0.268657f;
                    }
                } else {
                    if (features[5] <= -0.060745f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.477707f;
                        votes[1] += 0.012739f;
                        votes[3] += 0.070064f;
                        votes[4] += 0.038217f;
                        votes[5] += 0.401274f;
                    }
                }
            } else {
                if (features[0] <= -0.069581f) {
                    if (features[7] <= -3.638533f) {
                        votes[0] += 0.280000f;
                        votes[1] += 0.260000f;
                        votes[2] += 0.080000f;
                        votes[3] += 0.060000f;
                        votes[4] += 0.100000f;
                        votes[5] += 0.220000f;
                    } else {
                        votes[3] += 0.540000f;
                        votes[4] += 0.460000f;
                    }
                } else {
                    if (features[7] <= 0.363472f) {
                        votes[0] += 0.005096f;
                        votes[1] += 0.988535f;
                        votes[3] += 0.001115f;
                        votes[4] += 0.000955f;
                        votes[5] += 0.004299f;
                    } else {
                        votes[0] += 0.183673f;
                        votes[1] += 0.459184f;
                        votes[3] += 0.142857f;
                        votes[4] += 0.122449f;
                        votes[5] += 0.091837f;
                    }
                }
            }
        } else {
            if (features[2] <= 0.062091f) {
                if (features[1] <= 1.557840f) {
                    if (features[6] <= 0.298236f) {
                        votes[0] += 0.163353f;
                        votes[1] += 0.163353f;
                        votes[2] += 0.124127f;
                        votes[3] += 0.205266f;
                        votes[4] += 0.173025f;
                        votes[5] += 0.170876f;
                    } else {
                        votes[0] += 0.302846f;
                        votes[1] += 0.168699f;
                        votes[3] += 0.168699f;
                        votes[4] += 0.121951f;
                        votes[5] += 0.237805f;
                    }
                } else {
                    if (features[4] <= -1.476467f) {
                        votes[0] += 0.133028f;
                        votes[1] += 0.169725f;
                        votes[2] += 0.174312f;
                        votes[3] += 0.178899f;
                        votes[4] += 0.178899f;
                        votes[5] += 0.165138f;
                    } else {
                        votes[0] += 0.023904f;
                        votes[1] += 0.083665f;
                        votes[2] += 0.812749f;
                        votes[3] += 0.047809f;
                        votes[4] += 0.031873f;
                    }
                }
            } else {
                if (features[2] <= 0.358211f) {
                    if (features[3] <= 0.985071f) {
                        votes[0] += 0.000649f;
                        votes[1] += 0.000162f;
                        votes[2] += 0.997567f;
                        votes[3] += 0.001135f;
                        votes[4] += 0.000162f;
                        votes[5] += 0.000324f;
                    } else {
                        votes[0] += 0.155405f;
                        votes[1] += 0.152027f;
                        votes[2] += 0.320946f;
                        votes[3] += 0.111486f;
                        votes[4] += 0.125000f;
                        votes[5] += 0.135135f;
                    }
                } else {
                    if (features[2] <= 0.551027f) {
                        votes[0] += 0.096491f;
                        votes[1] += 0.105263f;
                        votes[2] += 0.333333f;
                        votes[3] += 0.166667f;
                        votes[4] += 0.192982f;
                        votes[5] += 0.105263f;
                    } else {
                        votes[0] += 0.337662f;
                        votes[1] += 0.142857f;
                        votes[2] += 0.077922f;
                        votes[3] += 0.155844f;
                        votes[4] += 0.090909f;
                        votes[5] += 0.194805f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_25(const float features[8], float votes[6]) {
    if (features[5] <= -0.095281f) {
        if (features[1] <= -0.281721f) {
            if (features[3] <= -0.422294f) {
                if (features[5] <= -0.180815f) {
                    if (features[5] <= -6.568605f) {
                        votes[0] += 0.193939f;
                        votes[1] += 0.127273f;
                        votes[2] += 0.072727f;
                        votes[3] += 0.133333f;
                        votes[4] += 0.369697f;
                        votes[5] += 0.103030f;
                    } else {
                        votes[0] += 0.001957f;
                        votes[3] += 0.001957f;
                        votes[4] += 0.992172f;
                        votes[5] += 0.003914f;
                    }
                } else {
                    if (features[2] <= -0.057887f) {
                        votes[3] += 1.000000f;
                    } else {
                        votes[0] += 0.500000f;
                        votes[5] += 0.500000f;
                    }
                }
            } else {
                if (features[5] <= -0.170007f) {
                    if (features[1] <= -0.340373f) {
                        votes[0] += 0.005243f;
                        votes[3] += 0.006641f;
                        votes[4] += 0.981475f;
                        votes[5] += 0.006641f;
                    } else {
                        votes[0] += 0.116667f;
                        votes[3] += 0.050000f;
                        votes[4] += 0.616667f;
                        votes[5] += 0.216667f;
                    }
                } else {
                    if (features[2] <= 0.300270f) {
                        votes[0] += 0.352518f;
                        votes[3] += 0.100719f;
                        votes[4] += 0.165468f;
                        votes[5] += 0.381295f;
                    } else {
                        votes[0] += 0.001357f;
                        votes[3] += 0.004071f;
                        votes[4] += 0.990502f;
                        votes[5] += 0.004071f;
                    }
                }
            }
        } else {
            if (features[1] <= -0.026047f) {
                if (features[5] <= -0.207735f) {
                    if (features[2] <= 0.295245f) {
                        votes[0] += 0.097561f;
                        votes[1] += 0.097561f;
                        votes[2] += 0.073171f;
                        votes[3] += 0.121951f;
                        votes[4] += 0.512195f;
                        votes[5] += 0.097561f;
                    } else {
                        votes[0] += 0.285714f;
                        votes[1] += 0.619048f;
                        votes[5] += 0.095238f;
                    }
                } else {
                    if (features[3] <= -0.069835f) {
                        votes[1] += 1.000000f;
                    } else {
                        votes[0] += 0.083333f;
                        votes[3] += 0.166667f;
                        votes[4] += 0.250000f;
                        votes[5] += 0.500000f;
                    }
                }
            } else {
                if (features[2] <= 0.051905f) {
                    if (features[0] <= -0.140751f) {
                        votes[0] += 0.163540f;
                        votes[1] += 0.157423f;
                        votes[2] += 0.182708f;
                        votes[3] += 0.167210f;
                        votes[4] += 0.168842f;
                        votes[5] += 0.160277f;
                    } else {
                        votes[1] += 1.000000f;
                    }
                } else {
                    if (features[7] <= 0.366265f) {
                        votes[0] += 0.013459f;
                        votes[1] += 0.006729f;
                        votes[2] += 0.950202f;
                        votes[4] += 0.020188f;
                        votes[5] += 0.009421f;
                    } else {
                        votes[0] += 0.057143f;
                        votes[3] += 0.142857f;
                        votes[4] += 0.714286f;
                        votes[5] += 0.085714f;
                    }
                }
            }
        }
    } else {
        if (features[1] <= -0.226573f) {
            if (features[2] <= -0.039135f) {
                if (features[4] <= 0.732541f) {
                    votes[4] += 1.000000f;
                } else {
                    votes[3] += 1.000000f;
                }
            } else {
                if (features[2] <= 0.122498f) {
                    if (features[5] <= 0.255091f) {
                        votes[0] += 0.459916f;
                        votes[5] += 0.540084f;
                    } else {
                        votes[0] += 0.505757f;
                        votes[5] += 0.494243f;
                    }
                } else {
                    if (features[5] <= 0.142259f) {
                        votes[0] += 0.256590f;
                        votes[3] += 0.101470f;
                        votes[4] += 0.394915f;
                        votes[5] += 0.247026f;
                    } else {
                        votes[0] += 0.311616f;
                        votes[1] += 0.000149f;
                        votes[3] += 0.355994f;
                        votes[4] += 0.025316f;
                        votes[5] += 0.306925f;
                    }
                }
            }
        } else {
            if (features[3] <= 0.597206f) {
                if (features[2] <= -0.044831f) {
                    votes[3] += 1.000000f;
                } else {
                    if (features[3] <= -0.059752f) {
                        votes[0] += 0.000515f;
                        votes[1] += 0.998970f;
                        votes[3] += 0.000172f;
                        votes[5] += 0.000343f;
                    } else {
                        votes[0] += 0.212389f;
                        votes[1] += 0.442478f;
                        votes[3] += 0.061947f;
                        votes[4] += 0.044248f;
                        votes[5] += 0.238938f;
                    }
                }
            } else {
                if (features[3] <= 0.888154f) {
                    if (features[6] <= -0.927712f) {
                        votes[0] += 0.428571f;
                        votes[5] += 0.571429f;
                    } else {
                        votes[0] += 0.001979f;
                        votes[1] += 0.002519f;
                        votes[2] += 0.991182f;
                        votes[3] += 0.001440f;
                        votes[4] += 0.000180f;
                        votes[5] += 0.002699f;
                    }
                } else {
                    if (features[5] <= 4.936506f) {
                        votes[0] += 0.152760f;
                        votes[1] += 0.193838f;
                        votes[2] += 0.272144f;
                        votes[3] += 0.168164f;
                        votes[4] += 0.047497f;
                        votes[5] += 0.165597f;
                    } else {
                        votes[2] += 0.130435f;
                        votes[3] += 0.130435f;
                        votes[4] += 0.739130f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_26(const float features[8], float votes[6]) {
    if (features[1] <= -0.226738f) {
        if (features[3] <= -0.334749f) {
            if (features[5] <= -0.022825f) {
                if (features[7] <= -0.014051f) {
                    if (features[6] <= 0.758120f) {
                        votes[0] += 0.033033f;
                        votes[1] += 0.018018f;
                        votes[2] += 0.045045f;
                        votes[3] += 0.036036f;
                        votes[4] += 0.840841f;
                        votes[5] += 0.027027f;
                    } else {
                        votes[0] += 0.358650f;
                        votes[3] += 0.337553f;
                        votes[5] += 0.303797f;
                    }
                } else {
                    if (features[5] <= -0.123402f) {
                        votes[0] += 0.008895f;
                        votes[3] += 0.010483f;
                        votes[4] += 0.973316f;
                        votes[5] += 0.007306f;
                    } else {
                        votes[0] += 0.078714f;
                        votes[3] += 0.032151f;
                        votes[4] += 0.807095f;
                        votes[5] += 0.082040f;
                    }
                }
            } else {
                if (features[2] <= -0.024424f) {
                    if (features[3] <= -0.365483f) {
                        votes[3] += 0.789100f;
                        votes[4] += 0.210900f;
                    } else {
                        votes[3] += 0.996117f;
                        votes[4] += 0.003883f;
                    }
                } else {
                    if (features[1] <= -0.472406f) {
                        votes[0] += 0.345836f;
                        votes[3] += 0.268505f;
                        votes[4] += 0.031440f;
                        votes[5] += 0.354220f;
                    } else {
                        votes[0] += 0.393394f;
                        votes[3] += 0.017546f;
                        votes[4] += 0.200383f;
                        votes[5] += 0.388676f;
                    }
                }
            }
        } else {
            if (features[4] <= -0.881912f) {
                if (features[2] <= 0.201355f) {
                    votes[4] += 1.000000f;
                } else {
                    votes[3] += 1.000000f;
                }
            } else {
                if (features[0] <= 0.297988f) {
                    if (features[7] <= 0.357023f) {
                        votes[0] += 0.310616f;
                        votes[3] += 0.094364f;
                        votes[4] += 0.318480f;
                        votes[5] += 0.276540f;
                    } else {
                        votes[0] += 0.083333f;
                        votes[3] += 0.800000f;
                        votes[4] += 0.033333f;
                        votes[5] += 0.083333f;
                    }
                } else {
                    if (features[4] <= 1.135104f) {
                        votes[0] += 0.003745f;
                        votes[3] += 0.978777f;
                        votes[4] += 0.015605f;
                        votes[5] += 0.001873f;
                    } else {
                        votes[0] += 0.085185f;
                        votes[1] += 0.002469f;
                        votes[3] += 0.723457f;
                        votes[4] += 0.109877f;
                        votes[5] += 0.079012f;
                    }
                }
            }
        }
    } else {
        if (features[1] <= 0.354382f) {
            if (features[3] <= -0.101466f) {
                if (features[0] <= 0.346416f) {
                    if (features[0] <= 0.191319f) {
                        votes[0] += 0.001850f;
                        votes[1] += 0.994450f;
                        votes[5] += 0.003700f;
                    } else {
                        votes[1] += 1.000000f;
                    }
                } else {
                    if (features[6] <= 1.252070f) {
                        votes[1] += 1.000000f;
                    } else {
                        votes[1] += 0.692308f;
                        votes[4] += 0.307692f;
                    }
                }
            } else {
                if (features[3] <= 0.229643f) {
                    if (features[0] <= -0.038889f) {
                        votes[0] += 0.117647f;
                        votes[1] += 0.102941f;
                        votes[2] += 0.014706f;
                        votes[3] += 0.367647f;
                        votes[4] += 0.338235f;
                        votes[5] += 0.058824f;
                    } else {
                        votes[0] += 0.282828f;
                        votes[1] += 0.333333f;
                        votes[3] += 0.070707f;
                        votes[4] += 0.070707f;
                        votes[5] += 0.242424f;
                    }
                } else {
                    if (features[5] <= -0.009964f) {
                        votes[0] += 0.027027f;
                        votes[1] += 0.297297f;
                        votes[2] += 0.054054f;
                        votes[3] += 0.054054f;
                        votes[4] += 0.405405f;
                        votes[5] += 0.162162f;
                    } else {
                        votes[0] += 0.166667f;
                        votes[1] += 0.516667f;
                        votes[3] += 0.016667f;
                        votes[4] += 0.016667f;
                        votes[5] += 0.283333f;
                    }
                }
            }
        } else {
            if (features[3] <= 0.794793f) {
                if (features[5] <= -0.678095f) {
                    if (features[0] <= -0.309454f) {
                        votes[0] += 0.183784f;
                        votes[1] += 0.102703f;
                        votes[2] += 0.086486f;
                        votes[3] += 0.270270f;
                        votes[4] += 0.162162f;
                        votes[5] += 0.194595f;
                    } else {
                        votes[2] += 1.000000f;
                    }
                } else {
                    if (features[3] <= 0.597206f) {
                        votes[1] += 0.583333f;
                        votes[3] += 0.083333f;
                        votes[4] += 0.333333f;
                    } else {
                        votes[0] += 0.002656f;
                        votes[1] += 0.000996f;
                        votes[2] += 0.990041f;
                        votes[3] += 0.001992f;
                        votes[4] += 0.000996f;
                        votes[5] += 0.003320f;
                    }
                }
            } else {
                if (features[1] <= 1.323644f) {
                    if (features[3] <= 1.766372f) {
                        votes[0] += 0.152111f;
                        votes[1] += 0.179039f;
                        votes[2] += 0.118632f;
                        votes[3] += 0.219796f;
                        votes[4] += 0.184862f;
                        votes[5] += 0.145560f;
                    } else {
                        votes[0] += 0.130000f;
                        votes[1] += 0.145000f;
                        votes[2] += 0.065000f;
                        votes[3] += 0.480000f;
                        votes[4] += 0.060000f;
                        votes[5] += 0.120000f;
                    }
                } else {
                    if (features[7] <= 0.534180f) {
                        votes[0] += 0.173751f;
                        votes[1] += 0.159582f;
                        votes[2] += 0.328859f;
                        votes[3] += 0.034303f;
                        votes[4] += 0.151380f;
                        votes[5] += 0.152125f;
                    } else {
                        votes[0] += 0.090909f;
                        votes[1] += 0.181818f;
                        votes[2] += 0.185771f;
                        votes[3] += 0.213439f;
                        votes[4] += 0.193676f;
                        votes[5] += 0.134387f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_27(const float features[8], float votes[6]) {
    if (features[2] <= 0.304686f) {
        if (features[5] <= -0.116277f) {
            if (features[7] <= -0.312620f) {
                if (features[7] <= -6.791758f) {
                    if (features[3] <= -1.326298f) {
                        votes[0] += 0.040404f;
                        votes[1] += 0.070707f;
                        votes[2] += 0.111111f;
                        votes[3] += 0.070707f;
                        votes[4] += 0.595960f;
                        votes[5] += 0.111111f;
                    } else {
                        votes[4] += 1.000000f;
                    }
                } else {
                    if (features[5] <= -0.793359f) {
                        votes[0] += 0.168831f;
                        votes[1] += 0.166334f;
                        votes[2] += 0.165335f;
                        votes[3] += 0.154845f;
                        votes[4] += 0.183816f;
                        votes[5] += 0.160839f;
                    } else {
                        votes[0] += 0.196154f;
                        votes[1] += 0.142308f;
                        votes[2] += 0.165385f;
                        votes[3] += 0.203846f;
                        votes[4] += 0.026923f;
                        votes[5] += 0.265385f;
                    }
                }
            } else {
                if (features[3] <= 0.523151f) {
                    if (features[7] <= -0.014051f) {
                        votes[0] += 0.127059f;
                        votes[1] += 0.057647f;
                        votes[3] += 0.068235f;
                        votes[4] += 0.629412f;
                        votes[5] += 0.117647f;
                    } else {
                        votes[0] += 0.017166f;
                        votes[1] += 0.019185f;
                        votes[3] += 0.010434f;
                        votes[4] += 0.938068f;
                        votes[5] += 0.015146f;
                    }
                } else {
                    if (features[3] <= 0.741958f) {
                        votes[0] += 0.012681f;
                        votes[2] += 0.945652f;
                        votes[3] += 0.010870f;
                        votes[4] += 0.018116f;
                        votes[5] += 0.012681f;
                    } else {
                        votes[0] += 0.100000f;
                        votes[1] += 0.124000f;
                        votes[2] += 0.200000f;
                        votes[3] += 0.108000f;
                        votes[4] += 0.340000f;
                        votes[5] += 0.128000f;
                    }
                }
            }
        } else {
            if (features[3] <= -0.294107f) {
                if (features[4] <= 0.974638f) {
                    if (features[0] <= 0.219131f) {
                        votes[3] += 0.891681f;
                        votes[4] += 0.108319f;
                    } else {
                        votes[0] += 0.296194f;
                        votes[3] += 0.292397f;
                        votes[4] += 0.114525f;
                        votes[5] += 0.296884f;
                    }
                } else {
                    if (features[7] <= 0.411230f) {
                        votes[0] += 0.388845f;
                        votes[3] += 0.122643f;
                        votes[4] += 0.091586f;
                        votes[5] += 0.396926f;
                    } else {
                        votes[0] += 0.003659f;
                        votes[3] += 0.996341f;
                    }
                }
            } else {
                if (features[5] <= 0.276824f) {
                    if (features[1] <= 0.878897f) {
                        votes[0] += 0.017978f;
                        votes[1] += 0.937079f;
                        votes[3] += 0.013162f;
                        votes[4] += 0.014125f;
                        votes[5] += 0.017657f;
                    } else {
                        votes[0] += 0.012559f;
                        votes[1] += 0.016641f;
                        votes[2] += 0.937520f;
                        votes[3] += 0.015385f;
                        votes[4] += 0.005024f;
                        votes[5] += 0.012873f;
                    }
                } else {
                    if (features[7] <= 0.330544f) {
                        votes[0] += 0.031488f;
                        votes[1] += 0.454545f;
                        votes[2] += 0.442357f;
                        votes[3] += 0.025901f;
                        votes[4] += 0.002539f;
                        votes[5] += 0.043169f;
                    } else {
                        votes[0] += 0.021840f;
                        votes[1] += 0.858883f;
                        votes[2] += 0.065099f;
                        votes[3] += 0.029399f;
                        votes[4] += 0.002520f;
                        votes[5] += 0.022260f;
                    }
                }
            }
        }
    } else {
        if (features[0] <= 0.304956f) {
            if (features[7] <= 0.015978f) {
                votes[2] += 1.000000f;
            } else {
                if (features[3] <= -0.338349f) {
                    if (features[1] <= -0.519153f) {
                        votes[0] += 0.545455f;
                        votes[3] += 0.090909f;
                        votes[5] += 0.363636f;
                    } else {
                        votes[0] += 0.012448f;
                        votes[3] += 0.009682f;
                        votes[4] += 0.958506f;
                        votes[5] += 0.019364f;
                    }
                } else {
                    if (features[5] <= -0.032759f) {
                        votes[2] += 1.000000f;
                    } else {
                        votes[0] += 0.007353f;
                        votes[1] += 0.882353f;
                        votes[2] += 0.014706f;
                        votes[3] += 0.058824f;
                        votes[4] += 0.007353f;
                        votes[5] += 0.029412f;
                    }
                }
            }
        } else {
            if (features[1] <= 0.828227f) {
                if (features[5] <= -0.089484f) {
                    if (features[6] <= 1.119098f) {
                        votes[0] += 0.500000f;
                        votes[5] += 0.500000f;
                    } else {
                        votes[0] += 0.024242f;
                        votes[1] += 0.054545f;
                        votes[2] += 0.006061f;
                        votes[3] += 0.018182f;
                        votes[4] += 0.860606f;
                        votes[5] += 0.036364f;
                    }
                } else {
                    if (features[7] <= 0.431272f) {
                        votes[0] += 0.244872f;
                        votes[1] += 0.280769f;
                        votes[3] += 0.100000f;
                        votes[4] += 0.132051f;
                        votes[5] += 0.242308f;
                    } else {
                        votes[0] += 0.107383f;
                        votes[1] += 0.053691f;
                        votes[3] += 0.751678f;
                        votes[4] += 0.033557f;
                        votes[5] += 0.053691f;
                    }
                }
            } else {
                if (features[1] <= 1.746309f) {
                    if (features[4] <= 1.412576f) {
                        votes[0] += 0.002068f;
                        votes[2] += 0.993108f;
                        votes[4] += 0.000689f;
                        votes[5] += 0.004135f;
                    } else {
                        votes[5] += 1.000000f;
                    }
                } else {
                    if (features[5] <= 0.087715f) {
                        votes[1] += 0.076923f;
                        votes[2] += 0.269231f;
                        votes[3] += 0.115385f;
                        votes[4] += 0.538462f;
                    } else {
                        votes[0] += 0.201149f;
                        votes[1] += 0.178161f;
                        votes[2] += 0.195402f;
                        votes[3] += 0.178161f;
                        votes[4] += 0.074713f;
                        votes[5] += 0.172414f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_28(const float features[8], float votes[6]) {
    if (features[3] <= -0.294102f) {
        if (features[7] <= 0.368921f) {
            if (features[0] <= -0.353816f) {
                if (features[5] <= -6.561912f) {
                    if (features[5] <= -6.821496f) {
                        votes[0] += 0.059829f;
                        votes[1] += 0.085470f;
                        votes[2] += 0.059829f;
                        votes[3] += 0.042735f;
                        votes[4] += 0.735043f;
                        votes[5] += 0.017094f;
                    } else {
                        votes[0] += 0.153846f;
                        votes[1] += 0.215385f;
                        votes[2] += 0.200000f;
                        votes[3] += 0.200000f;
                        votes[4] += 0.061538f;
                        votes[5] += 0.169231f;
                    }
                } else {
                    votes[4] += 1.000000f;
                }
            } else {
                if (features[2] <= -0.039135f) {
                    votes[3] += 1.000000f;
                } else {
                    if (features[1] <= -0.473350f) {
                        votes[0] += 0.280586f;
                        votes[3] += 0.211432f;
                        votes[4] += 0.222281f;
                        votes[5] += 0.285702f;
                    } else {
                        votes[0] += 0.342361f;
                        votes[1] += 0.000114f;
                        votes[3] += 0.021026f;
                        votes[4] += 0.287967f;
                        votes[5] += 0.348532f;
                    }
                }
            }
        } else {
            if (features[7] <= 0.497397f) {
                if (features[2] <= 0.338110f) {
                    if (features[2] <= 0.170337f) {
                        votes[0] += 0.100000f;
                        votes[3] += 0.300000f;
                        votes[5] += 0.600000f;
                    } else {
                        votes[3] += 1.000000f;
                    }
                } else {
                    if (features[5] <= 0.303689f) {
                        votes[4] += 1.000000f;
                    } else {
                        votes[0] += 0.400000f;
                        votes[5] += 0.600000f;
                    }
                }
            } else {
                votes[0] += 1.000000f;
            }
        }
    } else {
        if (features[7] <= -0.031283f) {
            if (features[1] <= 1.347146f) {
                if (features[2] <= -0.649343f) {
                    if (features[7] <= -4.007499f) {
                        votes[0] += 0.150510f;
                        votes[1] += 0.168367f;
                        votes[2] += 0.193878f;
                        votes[3] += 0.188776f;
                        votes[4] += 0.173469f;
                        votes[5] += 0.125000f;
                    } else {
                        votes[0] += 0.175137f;
                        votes[1] += 0.189210f;
                        votes[2] += 0.048475f;
                        votes[3] += 0.234558f;
                        votes[4] += 0.167318f;
                        votes[5] += 0.185301f;
                    }
                } else {
                    if (features[4] <= 0.864508f) {
                        votes[1] += 0.195652f;
                        votes[3] += 0.043478f;
                        votes[4] += 0.760870f;
                    } else {
                        votes[0] += 0.176471f;
                        votes[1] += 0.058824f;
                        votes[2] += 0.058824f;
                        votes[3] += 0.117647f;
                        votes[5] += 0.588235f;
                    }
                }
            } else {
                if (features[3] <= 1.482966f) {
                    if (features[0] <= -4.045687f) {
                        votes[0] += 0.162791f;
                        votes[1] += 0.232558f;
                        votes[2] += 0.162791f;
                        votes[3] += 0.093023f;
                        votes[4] += 0.209302f;
                        votes[5] += 0.139535f;
                    } else {
                        votes[0] += 0.040000f;
                        votes[1] += 0.030000f;
                        votes[2] += 0.930000f;
                    }
                } else {
                    if (features[3] <= 1.943839f) {
                        votes[0] += 0.187234f;
                        votes[1] += 0.234043f;
                        votes[2] += 0.110638f;
                        votes[3] += 0.042553f;
                        votes[4] += 0.170213f;
                        votes[5] += 0.255319f;
                    } else {
                        votes[0] += 0.167431f;
                        votes[1] += 0.155963f;
                        votes[2] += 0.321101f;
                        votes[3] += 0.045872f;
                        votes[4] += 0.158257f;
                        votes[5] += 0.151376f;
                    }
                }
            }
        } else {
            if (features[3] <= 0.597203f) {
                if (features[3] <= -0.156798f) {
                    if (features[1] <= -0.255296f) {
                        votes[0] += 0.287671f;
                        votes[3] += 0.191781f;
                        votes[4] += 0.210959f;
                        votes[5] += 0.309589f;
                    } else {
                        votes[1] += 1.000000f;
                    }
                } else {
                    if (features[1] <= -0.124058f) {
                        votes[0] += 0.177778f;
                        votes[1] += 0.007407f;
                        votes[3] += 0.318519f;
                        votes[4] += 0.251852f;
                        votes[5] += 0.244444f;
                    } else {
                        votes[0] += 0.106667f;
                        votes[1] += 0.635556f;
                        votes[3] += 0.106667f;
                        votes[4] += 0.062222f;
                        votes[5] += 0.088889f;
                    }
                }
            } else {
                if (features[1] <= 1.302078f) {
                    if (features[3] <= 0.786096f) {
                        votes[0] += 0.002598f;
                        votes[1] += 0.001624f;
                        votes[2] += 0.990581f;
                        votes[3] += 0.001949f;
                        votes[4] += 0.001624f;
                        votes[5] += 0.001624f;
                    } else {
                        votes[0] += 0.141844f;
                        votes[1] += 0.152482f;
                        votes[2] += 0.262411f;
                        votes[3] += 0.226950f;
                        votes[4] += 0.049645f;
                        votes[5] += 0.166667f;
                    }
                } else {
                    if (features[2] <= 0.620459f) {
                        votes[0] += 0.138978f;
                        votes[1] += 0.146965f;
                        votes[2] += 0.313099f;
                        votes[3] += 0.118211f;
                        votes[4] += 0.121406f;
                        votes[5] += 0.161342f;
                    } else {
                        votes[2] += 0.071429f;
                        votes[3] += 0.857143f;
                        votes[4] += 0.071429f;
                    }
                }
            }
        }
    }
}

static void evaluate_tree_29(const float features[8], float votes[6]) {
    if (features[0] <= 0.304686f) {
        if (features[1] <= -0.225485f) {
            if (features[0] <= -0.353816f) {
                if (features[6] <= -1.087752f) {
                    if (features[4] <= -2.781533f) {
                        votes[0] += 0.068376f;
                        votes[1] += 0.042735f;
                        votes[2] += 0.094017f;
                        votes[3] += 0.051282f;
                        votes[4] += 0.615385f;
                        votes[5] += 0.128205f;
                    } else {
                        votes[0] += 0.223881f;
                        votes[1] += 0.194030f;
                        votes[2] += 0.164179f;
                        votes[3] += 0.164179f;
                        votes[4] += 0.119403f;
                        votes[5] += 0.134328f;
                    }
                } else {
                    votes[4] += 1.000000f;
                }
            } else {
                if (features[2] <= -0.041490f) {
                    votes[3] += 1.000000f;
                } else {
                    if (features[5] <= -0.061799f) {
                        votes[0] += 0.061354f;
                        votes[3] += 0.013599f;
                        votes[4] += 0.864643f;
                        votes[5] += 0.060405f;
                    } else {
                        votes[0] += 0.321711f;
                        votes[3] += 0.268056f;
                        votes[4] += 0.088035f;
                        votes[5] += 0.322198f;
                    }
                }
            }
        } else {
            if (features[3] <= -0.069835f) {
                if (features[3] <= -0.144147f) {
                    votes[1] += 1.000000f;
                } else {
                    if (features[2] <= -0.052115f) {
                        votes[3] += 0.384615f;
                        votes[4] += 0.615385f;
                    } else {
                        votes[0] += 0.100000f;
                        votes[1] += 0.800000f;
                        votes[5] += 0.100000f;
                    }
                }
            } else {
                if (features[7] <= -0.029313f) {
                    if (features[1] <= 1.517613f) {
                        votes[0] += 0.194694f;
                        votes[1] += 0.171956f;
                        votes[2] += 0.107058f;
                        votes[3] += 0.167693f;
                        votes[4] += 0.181904f;
                        votes[5] += 0.176694f;
                    } else {
                        votes[0] += 0.077748f;
                        votes[1] += 0.115282f;
                        votes[2] += 0.514745f;
                        votes[3] += 0.075067f;
                        votes[4] += 0.104558f;
                        votes[5] += 0.112601f;
                    }
                } else {
                    if (features[7] <= 0.353506f) {
                        votes[0] += 0.018504f;
                        votes[1] += 0.020493f;
                        votes[2] += 0.912057f;
                        votes[3] += 0.019698f;
                        votes[4] += 0.009948f;
                        votes[5] += 0.019300f;
                    } else {
                        votes[0] += 0.147368f;
                        votes[1] += 0.161404f;
                        votes[2] += 0.217544f;
                        votes[3] += 0.150877f;
                        votes[4] += 0.133333f;
                        votes[5] += 0.189474f;
                    }
                }
            }
        }
    } else {
        if (features[0] <= 0.304972f) {
            if (features[5] <= -0.105182f) {
                votes[4] += 1.000000f;
            } else {
                if (features[4] <= 1.274054f) {
                    if (features[3] <= 0.208824f) {
                        votes[0] += 0.017094f;
                        votes[1] += 0.076923f;
                        votes[3] += 0.145299f;
                        votes[4] += 0.726496f;
                        votes[5] += 0.034188f;
                    } else {
                        votes[2] += 1.000000f;
                    }
                } else {
                    if (features[2] <= 0.304884f) {
                        votes[0] += 0.047244f;
                        votes[1] += 0.921260f;
                        votes[3] += 0.007874f;
                        votes[5] += 0.023622f;
                    } else {
                        votes[0] += 0.027027f;
                        votes[1] += 0.054054f;
                        votes[3] += 0.027027f;
                        votes[4] += 0.810811f;
                        votes[5] += 0.081081f;
                    }
                }
            }
        } else {
            if (features[3] <= 0.585575f) {
                if (features[1] <= -0.231045f) {
                    if (features[7] <= 0.431272f) {
                        votes[0] += 0.286285f;
                        votes[3] += 0.117177f;
                        votes[4] += 0.336884f;
                        votes[5] += 0.259654f;
                    } else {
                        votes[0] += 0.074074f;
                        votes[3] += 0.888889f;
                        votes[5] += 0.037037f;
                    }
                } else {
                    if (features[1] <= 0.124447f) {
                        votes[1] += 0.980769f;
                        votes[3] += 0.007692f;
                        votes[4] += 0.011538f;
                    } else {
                        votes[0] += 0.166667f;
                        votes[1] += 0.250000f;
                        votes[3] += 0.166667f;
                        votes[4] += 0.166667f;
                        votes[5] += 0.250000f;
                    }
                }
            } else {
                if (features[2] <= 0.358802f) {
                    if (features[7] <= 0.385800f) {
                        votes[2] += 1.000000f;
                    } else {
                        votes[0] += 0.258621f;
                        votes[1] += 0.155172f;
                        votes[2] += 0.241379f;
                        votes[3] += 0.068966f;
                        votes[4] += 0.086207f;
                        votes[5] += 0.189655f;
                    }
                } else {
                    if (features[3] <= 4.185061f) {
                        votes[0] += 0.149254f;
                        votes[1] += 0.164179f;
                        votes[2] += 0.402985f;
                        votes[3] += 0.119403f;
                        votes[4] += 0.074627f;
                        votes[5] += 0.089552f;
                    } else {
                        votes[0] += 0.135484f;
                        votes[1] += 0.277419f;
                        votes[2] += 0.064516f;
                        votes[3] += 0.096774f;
                        votes[4] += 0.206452f;
                        votes[5] += 0.219355f;
                    }
                }
            }
        }
    }
}


const char* get_fault_name(int fault_code) {
    if (fault_code >= 0 && fault_code < NUM_CLASSES) {
        return FAULT_NAMES[fault_code];
    }
    return "UNKNOWN_FAULT";
}

void get_class_probabilities(const float raw_features[NUM_FEATURES], float out_probs[NUM_CLASSES]) {
    float scaled[NUM_FEATURES];
    float votes[NUM_CLASSES] = {0.0f};
    
    // Step 1: Scale raw window features
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled[i] = (raw_features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
    
    // Step 2: Accumulate votes across all 30 trees

    evaluate_tree_0(scaled, votes);
    evaluate_tree_1(scaled, votes);
    evaluate_tree_2(scaled, votes);
    evaluate_tree_3(scaled, votes);
    evaluate_tree_4(scaled, votes);
    evaluate_tree_5(scaled, votes);
    evaluate_tree_6(scaled, votes);
    evaluate_tree_7(scaled, votes);
    evaluate_tree_8(scaled, votes);
    evaluate_tree_9(scaled, votes);
    evaluate_tree_10(scaled, votes);
    evaluate_tree_11(scaled, votes);
    evaluate_tree_12(scaled, votes);
    evaluate_tree_13(scaled, votes);
    evaluate_tree_14(scaled, votes);
    evaluate_tree_15(scaled, votes);
    evaluate_tree_16(scaled, votes);
    evaluate_tree_17(scaled, votes);
    evaluate_tree_18(scaled, votes);
    evaluate_tree_19(scaled, votes);
    evaluate_tree_20(scaled, votes);
    evaluate_tree_21(scaled, votes);
    evaluate_tree_22(scaled, votes);
    evaluate_tree_23(scaled, votes);
    evaluate_tree_24(scaled, votes);
    evaluate_tree_25(scaled, votes);
    evaluate_tree_26(scaled, votes);
    evaluate_tree_27(scaled, votes);
    evaluate_tree_28(scaled, votes);
    evaluate_tree_29(scaled, votes);

    // Step 3: Normalize to probability distribution
    float total_votes = 0.0f;
    for (int c = 0; c < NUM_CLASSES; c++) {
        total_votes += votes[c];
    }
    if (total_votes > 0.0f) {
        for (int c = 0; c < NUM_CLASSES; c++) {
            out_probs[c] = votes[c] / total_votes;
        }
    }
}

int predict_fault_raw_features(const float raw_features[NUM_FEATURES]) {
    float probs[NUM_CLASSES];
    get_class_probabilities(raw_features, probs);
    
    int max_idx = 0;
    float max_prob = probs[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (probs[c] > max_prob) {
            max_prob = probs[c];
            max_idx = c;
        }
    }
    return max_idx;
}

int predict_fault_scaled_features(const float scaled_features[NUM_FEATURES]) {
    float votes[NUM_CLASSES] = {0.0f};

    evaluate_tree_0(scaled_features, votes);
    evaluate_tree_1(scaled_features, votes);
    evaluate_tree_2(scaled_features, votes);
    evaluate_tree_3(scaled_features, votes);
    evaluate_tree_4(scaled_features, votes);
    evaluate_tree_5(scaled_features, votes);
    evaluate_tree_6(scaled_features, votes);
    evaluate_tree_7(scaled_features, votes);
    evaluate_tree_8(scaled_features, votes);
    evaluate_tree_9(scaled_features, votes);
    evaluate_tree_10(scaled_features, votes);
    evaluate_tree_11(scaled_features, votes);
    evaluate_tree_12(scaled_features, votes);
    evaluate_tree_13(scaled_features, votes);
    evaluate_tree_14(scaled_features, votes);
    evaluate_tree_15(scaled_features, votes);
    evaluate_tree_16(scaled_features, votes);
    evaluate_tree_17(scaled_features, votes);
    evaluate_tree_18(scaled_features, votes);
    evaluate_tree_19(scaled_features, votes);
    evaluate_tree_20(scaled_features, votes);
    evaluate_tree_21(scaled_features, votes);
    evaluate_tree_22(scaled_features, votes);
    evaluate_tree_23(scaled_features, votes);
    evaluate_tree_24(scaled_features, votes);
    evaluate_tree_25(scaled_features, votes);
    evaluate_tree_26(scaled_features, votes);
    evaluate_tree_27(scaled_features, votes);
    evaluate_tree_28(scaled_features, votes);
    evaluate_tree_29(scaled_features, votes);

    int max_idx = 0;
    float max_votes = votes[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (votes[c] > max_votes) {
            max_votes = votes[c];
            max_idx = c;
        }
    }
    return max_idx;
}
