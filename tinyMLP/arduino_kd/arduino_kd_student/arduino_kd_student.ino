#include <Arduino.h>
#include <TensorFlowLite.h>

#include "model_kd.h" 
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// =========================
// CONFIG
// =========================
static const int DEFAULT_REPS = 60000;
static const int N_ROUNDS = 10;
static const unsigned long PAUSE_TIME_MS = 60000UL;  // 60s

static const int CTX = 7;
static const int N_FEATURES = 5;
static const int INPUT_LEN = CTX * N_FEATURES;

// Vetor base (5 features) replicado pelos 7 timesteps
static const float DEFAULT_INPUT_VECTOR[N_FEATURES] = {
  30.8f, 76.0f, 1.2f, 4.5f, 0.0f
};

// =========================
// StandardScaler (35 valores)
// Ordem: flatten time-major: t0 f0..f4, t1 f0..f4, ..., t6 f0..f4
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

// Se AllocateTensors falhar, aumente para 16*1024 ou 20*1024
constexpr int kTensorArenaSize = 12 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// =========================
// Helpers
// =========================
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

void fill_input_tensor_float() {
  float xbuf[INPUT_LEN];
  build_normalized_input(xbuf);

  for (int i = 0; i < INPUT_LEN; i++) {
    input->data.f[i] = xbuf[i];
  }
}

float read_output_float() {
  return output->data.f[0];
}

void print_csv_header_once() {
  Serial.println("round,phase,reps,total_us,avg_us");
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  tflite::InitializeTarget();

  model = tflite::GetModel(g_tinymlp_kd);
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

  if (input->type != kTfLiteFloat32) {
    Serial.println("WARN_INPUT_NOT_FLOAT32");
  }

  if (output->type != kTfLiteFloat32) {
    Serial.println("WARN_OUTPUT_NOT_FLOAT32");
  }

  // Prepara input float uma vez
  fill_input_tensor_float();

  // Warm-up
  for (int i = 0; i < 10; i++) {
    interpreter->Invoke();
  }

  // Mostra uma saída teste
  if (interpreter->Invoke() == kTfLiteOk) {
    float y = read_output_float();
    Serial.print("TEST_OUTPUT_FLOAT=");
    Serial.println(y, 6);
  } else {
    Serial.println("ERR_TEST_INVOKE");
  }

  print_csv_header_once();
}

void loop() {
  static bool done = false;
  if (done) {
    delay(1000);
    return;
  }
  done = true;

  for (int round_idx = 1; round_idx <= N_ROUNDS; round_idx++) {
    Serial.print(round_idx);
    Serial.print(",rest,0,0,0\n");
    delay(PAUSE_TIME_MS);

    unsigned long t_start = micros();
    for (int r = 0; r < DEFAULT_REPS; r++) {
      if (interpreter->Invoke() != kTfLiteOk) {
        Serial.print(round_idx);
        Serial.print(",err,");
        Serial.print(r);
        Serial.print(",0,0\n");
        return;
      }
    }
    unsigned long t_end = micros();

    unsigned long dt_us = (t_end - t_start);
    double avg_us = (double)dt_us / (double)DEFAULT_REPS;

    Serial.print(round_idx);
    Serial.print(",infer,");
    Serial.print(DEFAULT_REPS);
    Serial.print(",");
    Serial.print(dt_us);
    Serial.print(",");
    Serial.println(avg_us, 3);
  }

  Serial.println("done,done,0,0,0");
}
