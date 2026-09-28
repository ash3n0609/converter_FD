/*
 * Microcontroller Integration Example (STM32 / TI C2000 / ESP32)
 * Demonstrates real-time ADC sampling, window feature extraction,
 * and fault classification.
 */

#include <stdio.h>
#include <math.h>
#include "fault_classifier.h"

// Ring buffer to hold 50 samples of Load Voltage and Load Current
static float v_buffer[WINDOW_SIZE];
static float i_buffer[WINDOW_SIZE];
static int buffer_index = 0;

void process_adc_sample(float v_sample, float i_sample) {
    v_buffer[buffer_index] = v_sample;
    i_buffer[buffer_index] = i_sample;
    buffer_index++;
    
    // Every WINDOW_SIZE samples (50 points), run feature extraction & inference
    if (buffer_index >= WINDOW_SIZE) {
        buffer_index = 0; // Reset index
        
        // Compute 8 Window Statistics
        float v_sum = 0.0f, v_sq_sum = 0.0f;
        float i_sum = 0.0f, i_sq_sum = 0.0f;
        float v_min = v_buffer[0], v_max = v_buffer[0];
        float i_min = i_buffer[0], i_max = i_buffer[0];
        
        for (int k = 0; k < WINDOW_SIZE; k++) {
            float v = v_buffer[k];
            float i = i_buffer[k];
            
            v_sum += v;
            v_sq_sum += v * v;
            if (v < v_min) v_min = v;
            if (v > v_max) v_max = v;
            
            i_sum += i;
            i_sq_sum += i * i;
            if (i < i_min) i_min = i;
            if (i > i_max) i_max = i;
        }
        
        float v_mean = v_sum / WINDOW_SIZE;
        float v_rms = sqrtf(v_sq_sum / WINDOW_SIZE);
        float v_p2p = v_max - v_min;
        
        // Variance for standard deviation
        float v_var = (v_sq_sum / WINDOW_SIZE) - (v_mean * v_mean);
        float v_std = sqrtf(v_var > 0.0f ? v_var : 0.0f);
        
        float i_mean = i_sum / WINDOW_SIZE;
        float i_rms = sqrtf(i_sq_sum / WINDOW_SIZE);
        float i_p2p = i_max - i_min;
        float i_var = (i_sq_sum / WINDOW_SIZE) - (i_mean * i_mean);
        float i_std = sqrtf(i_var > 0.0f ? i_var : 0.0f);
        
        float features[NUM_FEATURES] = {
            v_mean, v_std, v_rms, v_p2p,
            i_mean, i_std, i_rms, i_p2p
        };
        
        // Run inference
        int predicted_fault = predict_fault_raw_features(features);
        const char* fault_name = get_fault_name(predicted_fault);
        
        if (predicted_fault != FAULT_HEALTHY) {
            printf("[WARNING] Converter Fault Detected! Class %d: %s\n", predicted_fault, fault_name);
            // Trigger MCU PWM Shutdown / Relay Protection here
        } else {
            printf("[OK] Converter Operating Normally (Healthy)\n");
        }
    }
}

int main() {
    printf("=== MCU Fault Classifier Test Run ===\n");
    // Dummy sample input (Simulating 50 ADC voltage/current readings)
    for (int s = 0; s < WINDOW_SIZE; s++) {
        process_adc_sample(24.1f, 2.05f);
    }
    return 0;
}
