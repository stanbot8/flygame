// Headless brain server: runs the spiking brain for a remote body.
//
// Pair with nmfly-sim on another machine:
//   nmfly-brain-server --spec drosophila_full.brain --bind 0.0.0.0
//   nmfly-sim --headless --brain HOST:9100 --brain-timeout 60
//
// Runs in lockstep with the body: each sensory message advances the brain by
// --ms-per-exchange, then one motor command is sent back. Serves one body,
// then exits, so every run starts from a fresh brain.
//
// There is no authentication. The default bind address only accepts
// connections from this machine; use --bind on trusted networks only.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tcp.h"  // wire format, socket headers

#ifndef _WIN32
  #include <signal.h>
#endif

#include "brain.h"

using namespace mjgame;

namespace {

struct ServerConfig {
    std::string spec;
    std::string bind_addr = "127.0.0.1";
    int   port            = kDefaultPort;
    float forward         = 1.0f;
    float turn            = 0.0f;
    int   ms_per_exchange = 20;  // nmfly-sim exchanges every 20 ms by default
};

void PrintUsage(const char* prog) {
    printf("Usage: %s --spec FILE [options]\n\n", prog);
    printf("Options:\n");
    printf("  --spec FILE          .brain spec to simulate (required)\n");
    printf("  --bind ADDR          Address to listen on (default: 127.0.0.1)\n");
    printf("                       Other machines need 0.0.0.0 or a specific IP.\n");
    printf("                       No authentication: trusted networks only.\n");
    printf("  --port N             TCP port (default: %d)\n", kDefaultPort);
    printf("  --forward F          Walking intent, 0 to 1 (default: 1)\n");
    printf("  --turn T             Turning intent, -1 to 1 (default: 0)\n");
    printf("  --ms-per-exchange N  Brain ms per body exchange (default: 20)\n");
    printf("  --help               Show this help\n");
}

bool ParseArgs(int argc, char** argv, ServerConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        bool has_value = i + 1 < argc;
        if (strcmp(arg, "--spec") == 0 && has_value) cfg.spec = argv[++i];
        else if (strcmp(arg, "--bind") == 0 && has_value) cfg.bind_addr = argv[++i];
        else if (strcmp(arg, "--port") == 0 && has_value) cfg.port = std::atoi(argv[++i]);
        else if (strcmp(arg, "--forward") == 0 && has_value) cfg.forward = std::strtof(argv[++i], nullptr);
        else if (strcmp(arg, "--turn") == 0 && has_value) cfg.turn = std::strtof(argv[++i], nullptr);
        else if (strcmp(arg, "--ms-per-exchange") == 0 && has_value) cfg.ms_per_exchange = std::atoi(argv[++i]);
        else {
            PrintUsage(argv[0]);
            return false;
        }
    }
    if (cfg.spec.empty()) {
        fprintf(stderr, "--spec is required\n");
        return false;
    }
    if (cfg.port < 1 || cfg.port > 65535) {
        fprintf(stderr, "--port must be between 1 and 65535\n");
        return false;
    }
    if (cfg.forward < 0.0f || cfg.forward > 1.0f) {
        fprintf(stderr, "--forward must be between 0 and 1\n");
        return false;
    }
    if (cfg.turn < -1.0f || cfg.turn > 1.0f) {
        fprintf(stderr, "--turn must be between -1 and 1\n");
        return false;
    }
    if (cfg.ms_per_exchange < 1) {
        fprintf(stderr, "--ms-per-exchange must be at least 1\n");
        return false;
    }
    return true;
}

void CloseSocket(SocketType s) {
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

SocketType Listen(const ServerConfig& cfg) {
    addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* resolved = nullptr;
    if (getaddrinfo(cfg.bind_addr.c_str(), std::to_string(cfg.port).c_str(),
                    &hints, &resolved) != 0) {
        return kInvalidSocket;
    }
    SocketType s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == kInvalidSocket) {
        freeaddrinfo(resolved);
        return kInvalidSocket;
    }
#ifndef _WIN32
    // On Windows this flag would let another process take over the port.
    int reuse = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif
    bool ok = bind(s, resolved->ai_addr, static_cast<int>(resolved->ai_addrlen)) == 0 &&
              listen(s, 1) == 0;
    freeaddrinfo(resolved);
    if (!ok) {
        CloseSocket(s);
        return kInvalidSocket;
    }
    return s;
}

bool RecvAll(SocketType s, uint8_t* buf, size_t n) {
    size_t received = 0;
    while (received < n) {
        int r = recv(s, reinterpret_cast<char*>(buf + received),
                     static_cast<int>(n - received), 0);
        if (r <= 0) return false;
        received += static_cast<size_t>(r);
    }
    return true;
}

bool SendAll(SocketType s, const void* data, size_t n) {
    const char* p = static_cast<const char*>(data);
    size_t sent = 0;
    while (sent < n) {
        int r = send(s, p + sent, static_cast<int>(n - sent), 0);
        if (r <= 0) return false;
        sent += static_cast<size_t>(r);
    }
    return true;
}

bool SendMessage(SocketType s, uint32_t type, const void* payload, uint32_t size) {
    MsgHeader hdr;
    hdr.type = type;
    hdr.payload_size = size;
    return SendAll(s, &hdr, sizeof(hdr)) && (size == 0 || SendAll(s, payload, size));
}

}  // namespace

int main(int argc, char** argv) {
    ServerConfig cfg;
    if (!ParseArgs(argc, argv, cfg)) return 1;

#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#else
    signal(SIGPIPE, SIG_IGN);  // a lost body should end the run, not kill it
#endif

    nmfly::Brain brain;
    if (!brain.Init(cfg.spec)) {
        fprintf(stderr, "[brain-server] Could not load %s\n", cfg.spec.c_str());
        return 1;
    }
    printf("[brain-server] Brain ready: %zu neurons, %zu synapses\n",
           brain.NeuronCount(), brain.SynapseCount());

    SocketType listener = Listen(cfg);
    if (listener == kInvalidSocket) {
        fprintf(stderr, "[brain-server] Could not listen on %s:%d\n",
                cfg.bind_addr.c_str(), cfg.port);
        return 1;
    }
    printf("[brain-server] Listening on %s:%d (forward=%.2f turn=%.2f, %d ms per exchange)\n",
           cfg.bind_addr.c_str(), cfg.port, cfg.forward, cfg.turn, cfg.ms_per_exchange);
    fflush(stdout);

    SocketType client = accept(listener, nullptr, nullptr);
    CloseSocket(listener);
    if (client == kInvalidSocket) {
        fprintf(stderr, "[brain-server] Accept failed\n");
        return 1;
    }
    int nodelay = 1;
    setsockopt(client, IPPROTO_TCP, TCP_NODELAY,
               reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));
    printf("[brain-server] Body connected\n");
    fflush(stdout);

    nmfly::WasdIntent intent;
    intent.forward = cfg.forward;
    intent.turn = cfg.turn;

    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    long long exchanges = 0;
    long long total_spikes = 0;
    double total_forward = 0.0;
    std::vector<uint8_t> payload;
    std::vector<SensoryReading> sensory;
    int exit_code = 0;

    for (;;) {
        MsgHeader hdr;
        if (!RecvAll(client, reinterpret_cast<uint8_t*>(&hdr), sizeof(hdr))) break;
        if (hdr.payload_size > (1u << 20)) {
            fprintf(stderr, "[brain-server] Oversized message (%u bytes)\n", hdr.payload_size);
            exit_code = 1;
            break;
        }
        payload.resize(hdr.payload_size);
        if (hdr.payload_size > 0 && !RecvAll(client, payload.data(), hdr.payload_size)) break;

        if (hdr.type == kMsgHelloClient) {
            uint32_t version = kProtocolVersion;
            if (!SendMessage(client, kMsgHelloServer, &version, sizeof(version))) break;
        } else if (hdr.type == kMsgPing) {
            if (!SendMessage(client, kMsgPong, nullptr, 0)) break;
        } else if (hdr.type == kMsgBioReadings) {
            size_t count = payload.size() / sizeof(BioReading);
            sensory.resize(count);
            for (size_t i = 0; i < count; ++i) {
                BioReading reading;
                std::memcpy(&reading, payload.data() + i * sizeof(BioReading), sizeof(reading));
                sensory[i].channel = reading.neuron_idx;
                sensory[i].activation = reading.spike_prob;
            }

            // Same per-millisecond order as the embedded brain thread.
            for (int ms = 0; ms < cfg.ms_per_exchange; ++ms) {
                brain.SetProprio(sensory.data(), static_cast<int>(count));
                brain.SetIntent(intent);
                brain.Step(1.0f);
                total_spikes += brain.CountSpikes();
            }

            MotorCommand cmd = brain.GetMotorCommand();
            WireMotorCommand wire;
            wire.forward_velocity = cmd.forward_velocity;
            wire.angular_velocity = cmd.angular_velocity;
            wire.approach_drive   = cmd.approach_drive;
            wire.freeze           = cmd.freeze;
            if (!SendMessage(client, kMsgMotor, &wire, sizeof(wire))) break;

            ++exchanges;
            total_forward += cmd.forward_velocity;
            if (exchanges % 50 == 0) {
                printf("[brain-server] %lld exchanges, forward=%.2f, %lld spikes so far\n",
                       exchanges, cmd.forward_velocity, total_spikes);
                fflush(stdout);
            }
        }
    }

    double wall = std::chrono::duration<double>(Clock::now() - start).count();
    printf("[brain-server] Body gone. %lld exchanges, %lld spikes, mean forward=%.2f, "
           "%.1f ms wall per exchange\n",
           exchanges, total_spikes,
           exchanges > 0 ? total_forward / static_cast<double>(exchanges) : 0.0,
           exchanges > 0 ? wall * 1000.0 / static_cast<double>(exchanges) : 0.0);
    CloseSocket(client);
    return exit_code;
}
