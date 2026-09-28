/*
 * Fault Classifier for Buck-Boost DC-DC Converter
 * Automatically generated standalone MCU code.
 *
 * Target: 32-bit Microcontrollers (STM32, TI C2000, ESP32, ARM Cortex-M)
 * Memory Footprint: Flash < 30 KB, SRAM < 2 KB
 * Execution Time: < 50 microseconds @ 100MHz
 */

#ifndef FAULT_CLASSIFIER_H
#define FAULT_CLASSIFIER_H

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_FEATURES 8
#define NUM_CLASSES 6
#define WINDOW_SIZE 50

// Fault Class Enums
typedef enum {
    FAULT_HEALTHY = 0,
    FAULT_ESR_MILD = 1,
    FAULT_ESR_SEVERE = 2,
    FAULT_SWITCH_DEG = 3,
    FAULT_GATE_DEG = 4,
    FAULT_SENSOR_FAULT = 5
} FaultClass;

// Function Prototypes
const char* get_fault_name(int fault_code);
int predict_fault_raw_features(const float raw_features[NUM_FEATURES]);
int predict_fault_scaled_features(const float scaled_features[NUM_FEATURES]);
void get_class_probabilities(const float raw_features[NUM_FEATURES], float out_probs[NUM_CLASSES]);

#ifdef __cplusplus
}
#endif

#endif // FAULT_CLASSIFIER_H
