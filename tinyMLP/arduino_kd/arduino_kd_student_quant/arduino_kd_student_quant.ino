#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <TensorFlowLite.h>

#include "model_kd_ptq_int8.h"  

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// =========================
// CONFIG
// =========================
static const int DEFAULT_REPS = 60000;
static const int N_ROUNDS = 10;
static const unsigned long PAUSE_TIME_MS = 60000UL;   // 60s
static const unsigned long SAMPLE_PERIOD_MS = 20UL;   // amostragem do INA219

static const int CTX = 7;
static const int N_FEATURES = 5;
static const int INPUT_LEN = CTX * N_FEATURES;

// Vetor base (5 features) replicado pelos 7 timesteps
static const float DEFAULT_INPUT_VECTOR[N_FEATURES] = {
  30.8f, 76.0f, 1.2f, 4.5f, 0.0f
};

// =========================
// Quant params do KD PTQ int8
// =========================
static const float INPUT_SCALE = 0.054823506623506546f;
static const int INPUT_ZERO_POINT = -76;

static const float OUTPUT_SCALE = 0.17461146414279938f;
static const int OUTPUT_ZERO_POINT = -127;

// =========================
// StandardScaler
// =========================
static const float SCALER_MEAN[INPUT_LEN] = {
  27.37125811277017,
    85.217659137577,
    0.9478661441265117,
    2.3300627374286504,
    1.1489705283413307,
    27.371291410976593,
    85.21793662245408,
    0.9477440507779528,
    2.329840749526991,
    1.1489372301547587,
    27.37132470918301,
    85.21821410733115,
    0.9476219574293938,
    2.3296187616253317,
    1.1489039319681866,
    27.371413504364845,
    85.21832510128198,
    0.9476219574293938,
    2.3296187616253317,
    1.1488706337816146,
    27.37154114736596,
    85.2183805982574,
    0.9476219574293938,
    2.3295410658610742,
    1.1488206865050643,
    27.3716576910355,
    85.2184360952328,
    0.9476219574293938,
    2.3294800191867946,
    1.148770739228514,
    27.371774234705043,
    85.21849159220822,
    0.9476219574293938,
    2.3294800191867946,
    1.148720791951964
};

static const float SCALER_SCALE[INPUT_LEN] = {
  2.470521081568853,
    9.825432129733988,
    1.2184702853808778,
    2.2408775470575524,
    3.9985249027418712,
    2.47048642514252,
    9.825750750805543,
    1.2184550350256347,
    2.2409102328061636,
    3.998531972498796,
    2.4704517677811326,
    9.826069353708524,
    1.2184397722456315,
    2.2409428960882085,
    3.998539041965865,
    2.470341371432348,
    9.826213733346105,
    1.2184397722456317,
    2.2409428960882085,
    3.998546111143103,
    2.470200748167309,
    9.826288745802962,
    1.2184397722456317,
    2.2409612570855684,
    3.998554840666105,
    2.47007234023877,
    9.826363757373482,
    1.2184397722456317,
    2.2410097327229996,
    3.998563569546478,
    2.4699392016388115,
    9.826438768058162,
    1.2184397722456317,
    2.2410097327229996,
    3.998572297783123
};

// =========================
// TFLite Micro
// =========================
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

tflite::AllOpsResolver resolver;

constexpr int kTensorArenaSize = 12 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// =========================
// INA219
// =========================
Adafruit_INA219 ina219;
bool ina_ok = false;

// =========================
// Tempo global
// =========================
unsigned long global_t0_ms = 0;

// =========================
// Helpers
// =========================
int8_t clamp_int8(int v) {
  if (v > 127) return 127;
  if (v < -128) return -128;
  return (int8_t)v;
}

void build_normalized_input(float out_x[INPUT_LEN]) {
  int idx = 0;
  for (int t = 0; t < CTX; t++) {
    for (int f = 0; f < N_FEATURES; f++) {
      float v = DEFAULT_INPUT_VECTOR[f];
      v = (v - SCALER_MEAN[idx]) / SCALER_SCALE[idx];
      out_x[idx] = v;
      idx++;
    }
  }
}

void quantize_and_fill_input_tensor() {
  float xbuf[INPUT_LEN];
  build_normalized_input(xbuf);

  for (int i = 0; i < INPUT_LEN; i++) {
    int q = (int) roundf(xbuf[i] / INPUT_SCALE + INPUT_ZERO_POINT);
    input->data.int8[i] = clamp_int8(q);
  }
}

float dequantize_output_value() {
  int8_t y_q = output->data.int8[0];
  return ((int)y_q - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
}

double now_seconds() {
  return (millis() - global_t0_ms) / 1000.0;
}

void print_power_sample(int round_idx, const char* phase, int inferindo) {
  if (!ina_ok) return;

  float busVoltage_V = ina219.getBusVoltage_V();
  float current_mA   = ina219.getCurrent_mA();
  float power_mW     = ina219.getPower_mW();

  Serial.print(now_seconds(), 3);
  Serial.print(",");
  Serial.print(power_mW / 1000.0, 6);
  Serial.print(",");
  Serial.print(inferindo);
  Serial.print(",");
  Serial.print(round_idx);
  Serial.print(",");
  Serial.print(phase);
  Serial.print(",");
  Serial.print(busVoltage_V, 4);
  Serial.print(",");
  Serial.println(current_mA, 3);
}

void log_rest_block(int round_idx, unsigned long duration_ms) {
  unsigned long t0 = millis();
  unsigned long last_sample = 0;

  while ((millis() - t0) < duration_ms) {
    unsigned long elapsed = millis() - t0;
    if (elapsed - last_sample >= SAMPLE_PERIOD_MS) {
      last_sample = elapsed;
      print_power_sample(round_idx, "rest", 0);
    }
  }
}

void print_summary_line(int round_idx, int reps, unsigned long total_us) {
  double avg_us = (reps > 0) ? ((double)total_us / (double)reps) : 0.0;

  Serial.print("SUMMARY,");
  Serial.print(round_idx);
  Serial.print(",infer,");
  Serial.print(reps);
  Serial.print(",");
  Serial.print(total_us);
  Serial.print(",");
  Serial.println(avg_us, 3);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  Wire.begin();

  if (ina219.begin()) {
    ina_ok = true;
  } else {
    Serial.println("ERR_INA219_NOT_FOUND");
  }

  tflite::InitializeTarget();

  model = tflite::GetModel(g_tinymlp_kd_ptq_int8);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERR_SCHEMA");
    while (1) {}
  }

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERR_ALLOC_TENSORS");
    while (1) {}
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("INPUT_TYPE=");
  Serial.println(input->type);

  Serial.print("OUTPUT_TYPE=");
  Serial.println(output->type);

  if (input->type != kTfLiteInt8) {
    Serial.println("WARN_INPUT_NOT_INT8");
  }

  if (output->type != kTfLiteInt8) {
    Serial.println("WARN_OUTPUT_NOT_INT8");
  }

  quantize_and_fill_input_tensor();

  for (int i = 0; i < 10; i++) {
    interpreter->Invoke();
  }

  if (interpreter->Invoke() == kTfLiteOk) {
    float y = dequantize_output_value();
    Serial.print("TEST_OUTPUT_FLOAT=");
    Serial.println(y, 6);
  } else {
    Serial.println("ERR_TEST_INVOKE");
  }

  global_t0_ms = millis();

  Serial.println("tempo_s,power_W,inferindo,round,phase,bus_V,current_mA");
  Serial.println("SUMMARY_HEADER,round,phase,reps,total_us,avg_us");
}

void loop() {
  static bool done = false;
  if (done) {
    delay(1000);
    return;
  }
  done = true;

  for (int round_idx = 1; round_idx <= N_ROUNDS; round_idx++) {
    log_rest_block(round_idx, PAUSE_TIME_MS);

    unsigned long infer_t0_us = micros();
    unsigned long last_sample_ms = millis();

    for (int r = 0; r < DEFAULT_REPS; r++) {
      if (interpreter->Invoke() != kTfLiteOk) {
        Serial.print("ERR_INVOKE_ROUND=");
        Serial.print(round_idx);
        Serial.print(",REP=");
        Serial.println(r);
        return;
      }

      unsigned long now_ms = millis();
      if (now_ms - last_sample_ms >= SAMPLE_PERIOD_MS) {
        last_sample_ms = now_ms;
        print_power_sample(round_idx, "infer", 1);
      }
    }

    unsigned long infer_t1_us = micros();
    unsigned long dt_us = infer_t1_us - infer_t0_us;

    print_power_sample(round_idx, "infer", 1);
    print_summary_line(round_idx, DEFAULT_REPS, dt_us);
  }

  Serial.println("DONE");
}
