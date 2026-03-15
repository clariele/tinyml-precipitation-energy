#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <TensorFlowLite.h>

#include "model_ptq_int8.h"  

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// =========================================================
// CONFIG
// =========================================================
static const int DEFAULT_REPS = 60000;
static const int N_ROUNDS = 10;
static const unsigned long PAUSE_TIME_MS = 60000UL;   // 60 s
static const unsigned long SAMPLE_PERIOD_MS = 100UL;  // amostragem de potência

static const int CTX = 7;
static const int N_FEATURES = 5;
static const int INPUT_LEN = CTX * N_FEATURES;

// Vetor base (5 features) replicado pelos 7 timesteps
static const float DEFAULT_INPUT_VECTOR[N_FEATURES] = {
  30.8f, 76.0f, 1.2f, 4.5f, 0.0f
};

// =========================================================
// StandardScaler (35 valores)
// Ordem: flatten time-major: t0 f0..f4, t1 f0..f4, ..., t6 f0..f4
// =========================================================
static const float SCALER_MEAN[INPUT_LEN] = {
  27.37125811277017f,
  85.217659137577f,
  0.9478661441265117f,
  2.3300627374286504f,
  1.1489705283413307f,
  27.371291410976593f,
  85.21793662245408f,
  0.9477440507779528f,
  2.329840749526991f,
  1.1489372301547587f,
  27.37132470918301f,
  85.21821410733115f,
  0.9476219574293938f,
  2.3296187616253317f,
  1.1489039319681866f,
  27.371413504364845f,
  85.21832510128198f,
  0.9476219574293938f,
  2.3296187616253317f,
  1.1488706337816146f,
  27.37154114736596f,
  85.2183805982574f,
  0.9476219574293938f,
  2.3295410658610742f,
  1.1488206865050643f,
  27.3716576910355f,
  85.2184360952328f,
  0.9476219574293938f,
  2.3294800191867946f,
  1.148770739228514f,
  27.371774234705043f,
  85.21849159220822f,
  0.9476219574293938f,
  2.3294800191867946f,
  1.148720791951964f
};

static const float SCALER_SCALE[INPUT_LEN] = {
  2.470521081568853f,
  9.825432129733988f,
  1.2184702853808778f,
  2.2408775470575524f,
  3.9985249027418712f,
  2.47048642514252f,
  9.825750750805543f,
  1.2184550350256347f,
  2.2409102328061636f,
  3.998531972498796f,
  2.4704517677811326f,
  9.826069353708524f,
  1.2184397722456315f,
  2.2409428960882085f,
  3.998539041965865f,
  2.470341371432348f,
  9.826213733346105f,
  1.2184397722456317f,
  2.2409428960882085f,
  3.998546111143103f,
  2.470200748167309f,
  9.826288745802962f,
  1.2184397722456317f,
  2.2409612570855684f,
  3.998554840666105f,
  2.47007234023877f,
  9.826363757373482f,
  1.2184397722456317f,
  2.2410097327229996f,
  3.998563569546478f,
  2.4699392016388115f,
  9.826438768058162f,
  1.2184397722456317f,
  2.2410097327229996f,
  3.998572297783123f
};

// =========================================================
// QUANT PARAMS DO MODELO INT8
// quant_params_ptq_int8.json
// =========================================================
static const float INPUT_QUANT_SCALE = 0.054823506623506546f;
static const int   INPUT_QUANT_ZERO_POINT = -76;

static const float OUTPUT_QUANT_SCALE = 0.17296575009822845f;
static const int   OUTPUT_QUANT_ZERO_POINT = -122;

// =========================================================
// TFLite Micro
// =========================================================
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

tflite::AllOpsResolver resolver;

// Se AllocateTensors falhar, aumente
constexpr int kTensorArenaSize = 16 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// =========================================================
// INA219
// =========================================================
Adafruit_INA219 ina219;

// =========================================================
// TEMPO GLOBAL
// =========================================================
unsigned long global_start_us = 0;

// =========================================================
// HELPERS
// =========================================================
void print_power_csv_header_once() {
  Serial.println("tempo_s,power_W,inferindo,round,phase,bus_V,current_mA");
}

void print_summary_line(int round_idx, const char* phase, int reps, unsigned long total_us, double avg_us) {
  Serial.print("SUMMARY,");
  Serial.print(round_idx);
  Serial.print(",");
  Serial.print(phase);
  Serial.print(",");
  Serial.print(reps);
  Serial.print(",");
  Serial.print(total_us);
  Serial.print(",");
  Serial.println(avg_us, 3);
}

void sample_and_print_power(int inferindo, int round_idx, const char* phase) {
  float bus_V = ina219.getBusVoltage_V();
  float current_mA = ina219.getCurrent_mA();
  float power_W = bus_V * (current_mA / 1000.0f);

  unsigned long now_us = micros();
  float tempo_s = (now_us - global_start_us) / 1000000.0f;

  Serial.print(tempo_s, 6);
  Serial.print(",");
  Serial.print(power_W, 6);
  Serial.print(",");
  Serial.print(inferindo);
  Serial.print(",");
  Serial.print(round_idx);
  Serial.print(",");
  Serial.print(phase);
  Serial.print(",");
  Serial.print(bus_V, 6);
  Serial.print(",");
  Serial.println(current_mA, 6);
}

int8_t quantize_to_int8(float x, float scale, int zero_point) {
  int32_t q = (int32_t)round(x / scale) + zero_point;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}

float dequantize_from_int8(int8_t q, float scale, int zero_point) {
  return ((int)q - zero_point) * scale;
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

void fill_input_tensor_int8() {
  float xbuf[INPUT_LEN];
  build_normalized_input(xbuf);

  // Shape lógico: (1, 7, 5)
  // Em memória: buffer linear de 35 int8
  for (int i = 0; i < INPUT_LEN; i++) {
    input->data.int8[i] = quantize_to_int8(
      xbuf[i],
      INPUT_QUANT_SCALE,
      INPUT_QUANT_ZERO_POINT
    );
  }
}

float read_output_float() {
  return dequantize_from_int8(
    output->data.int8[0],
    OUTPUT_QUANT_SCALE,
    OUTPUT_QUANT_ZERO_POINT
  );
}

// =========================================================
// MEDIÇÃO DE REPOUSO
// =========================================================
void measure_rest_phase(int round_idx) {
  unsigned long start_ms = millis();
  unsigned long last_sample_ms = 0;

  while ((millis() - start_ms) < PAUSE_TIME_MS) {
    unsigned long now_ms = millis();
    if ((now_ms - last_sample_ms) >= SAMPLE_PERIOD_MS) {
      sample_and_print_power(0, round_idx, "rest");
      last_sample_ms = now_ms;
    }
  }
}

// =========================================================
// MEDIÇÃO DE INFERÊNCIA
// =========================================================
void measure_infer_phase(int round_idx) {
  unsigned long last_sample_ms = millis();

  // amostra logo no início da inferência
  sample_and_print_power(1, round_idx, "infer");

  unsigned long t_start = micros();

  for (int r = 0; r < DEFAULT_REPS; r++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.print("SUMMARY,");
      Serial.print(round_idx);
      Serial.println(",err,0,0,0");
      return;
    }

    unsigned long now_ms = millis();
    if ((now_ms - last_sample_ms) >= SAMPLE_PERIOD_MS) {
      sample_and_print_power(1, round_idx, "infer");
      last_sample_ms = now_ms;
    }
  }

  unsigned long t_end = micros();
  unsigned long dt_us = (t_end - t_start);
  double avg_us = (double)dt_us / (double)DEFAULT_REPS;

  // garante amostra no final
  sample_and_print_power(1, round_idx, "infer");

  print_summary_line(round_idx, "infer", DEFAULT_REPS, dt_us, avg_us);
}

// =========================================================
// SETUP
// =========================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  Wire.begin();

  if (!ina219.begin()) {
    Serial.println("ERR_INA219_NOT_FOUND");
    while (1) { delay(1000); }
  }

  tflite::InitializeTarget();

  model = tflite::GetModel(g_tinycnn_ptq_int8);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERR_SCHEMA");
    while (1) { delay(1000); }
  }

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERR_ALLOC_TENSORS");
    while (1) { delay(1000); }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("INPUT_TYPE=");
  Serial.println(input->type);

  Serial.print("OUTPUT_TYPE=");
  Serial.println(output->type);

  Serial.print("INPUT_DIMS=");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print("x");
  }
  Serial.println();

  Serial.print("OUTPUT_DIMS=");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print("x");
  }
  Serial.println();

  Serial.print("INPUT_SCALE_MODEL=");
  Serial.println(input->params.scale, 10);
  Serial.print("INPUT_ZERO_POINT_MODEL=");
  Serial.println(input->params.zero_point);

  Serial.print("OUTPUT_SCALE_MODEL=");
  Serial.println(output->params.scale, 10);
  Serial.print("OUTPUT_ZERO_POINT_MODEL=");
  Serial.println(output->params.zero_point);

  if (input->type != kTfLiteInt8) {
    Serial.println("WARN_INPUT_NOT_INT8");
  }

  if (output->type != kTfLiteInt8) {
    Serial.println("WARN_OUTPUT_NOT_INT8");
  }

  // prepara input uma vez
  fill_input_tensor_int8();

  // warm-up
  for (int i = 0; i < 10; i++) {
    interpreter->Invoke();
  }

  // teste
  if (interpreter->Invoke() == kTfLiteOk) {
    float y = read_output_float();
    Serial.print("TEST_OUTPUT_FLOAT=");
    Serial.println(y, 6);
  } else {
    Serial.println("ERR_TEST_INVOKE");
  }

  global_start_us = micros();
  print_power_csv_header_once();
}

// =========================================================
// LOOP
// =========================================================
void loop() {
  static bool done = false;
  if (done) {
    delay(1000);
    return;
  }
  done = true;

  for (int round_idx = 1; round_idx <= N_ROUNDS; round_idx++) {
    measure_rest_phase(round_idx);
    measure_infer_phase(round_idx);
  }

  Serial.println("SUMMARY,done,done,0,0,0");
}
