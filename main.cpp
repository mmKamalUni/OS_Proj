#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <array>
#include <unistd.h>
#include <sys/wait.h>
#include <pthread.h>
#include <semaphore.h>

using namespace std;

struct NeuronArg {
    int id;
    const double* inputs;
    int input_size;
    double** weights;
    double* outputs;
    pthread_mutex_t* mutex;
};

double* readDoublesFromFile(const string& filename, int& total_size) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        total_size = 0;
        return nullptr;
    }
    vector<double> temp;
    string line;
    while (getline(file, line)) {
        // replace commas with spaces so stringstream parsing works
        for (char &c : line) if (c == ',') c = ' ';
        istringstream iss(line);
        double v;
        while (iss >> v) temp.push_back(v);
    }
    total_size = temp.size();
    double* array = new double[total_size];
    for (int i = 0; i < total_size; ++i) array[i] = temp[i];
    return array;
}

double** get_weights(int prev_neurons, int curr_neurons, const double* all_data, int& offset) {
    double** weights = new double*[curr_neurons];
    for (int i = 0; i < curr_neurons; ++i) {
        weights[i] = new double[prev_neurons];
        for (int j = 0; j < prev_neurons; ++j) {
            weights[i][j] = all_data[offset++];
        }
    }
    return weights;
}

void* compute_neuron(void* arg) {
    NeuronArg* narg = (NeuronArg*) arg;
    double sum = 0;
    for(int i = 0; i < narg->input_size; i++) {
        sum += narg->inputs[i] * narg->weights[narg->id][i];
    }
    pthread_mutex_lock(narg->mutex);
    narg->outputs[narg->id] = sum;
    pthread_mutex_unlock(narg->mutex);
    return nullptr;
}

double* compute_layer_outputs(int curr_neurons, int prev_neurons, const double* inputs, double** weights) {
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, nullptr);
    vector<pthread_t> threads(curr_neurons);
    double* outputs = new double[curr_neurons]();
    // allocate args on stack/heap container to avoid leaking per-thread allocations
    vector<NeuronArg> args(curr_neurons);
    for (int i = 0; i < curr_neurons; ++i) {
        args[i].id = i;
        args[i].inputs = inputs;
        args[i].input_size = prev_neurons;
        args[i].weights = weights;
        args[i].outputs = outputs;
        args[i].mutex = &mutex;
        pthread_create(&threads[i], nullptr, compute_neuron, &args[i]);
    }
    for (int i = 0; i < curr_neurons; ++i) pthread_join(threads[i], nullptr);
    pthread_mutex_destroy(&mutex);
    return outputs;
}

void write_to_pipe(int fd, const double* data, int size) {
    const char* buf = reinterpret_cast<const char*>(data);
    size_t to_write = sizeof(double) * size;
    size_t written = 0;
    while (written < to_write) {
        ssize_t r = write(fd, buf + written, to_write - written);
        if (r <= 0) {
            if (errno == EINTR) continue;
            perror("write");
            break;
        }
        written += r;
    }
}

void read_from_pipe(int fd, double* data, int size) {
    char* buf = reinterpret_cast<char*>(data);
    size_t to_read = sizeof(double) * size;
    size_t got = 0;
    while (got < to_read) {
        ssize_t r = read(fd, buf + got, to_read - got);
        if (r < 0) {
            if (errno == EINTR) continue;
            perror("read");
            break;
        }
        if (r == 0) {
            // EOF
            break;
        }
        got += r;
    }
    if (got != to_read) {
        // partial read — warn
        // do not perror here since errno may be 0
        cerr << "read: Partial read expected=" << to_read << " got=" << got << "\n";
    }
}

void process_layer(int layer_index, int prev_neurons, int curr_neurons, bool is_output, double* all_data, int read_fd, int write_fd, int read_back_fd, int write_back_fd, int offset_start) {
    double back[2];
    ofstream outfile("output.txt", ios::app);
    // Special-case input layer: pass-through of first two inputs
    if (layer_index == 0) {
        double* outputs = new double[2];
        outputs[0] = all_data[0];
        outputs[1] = all_data[1];
        cout << "Layer " << layer_index << " Forward Pass 1 outputs: " << outputs[0] << " " << outputs[1] << endl;
        outfile << outputs[0] << " " << outputs[1] << endl;
        if (!is_output && write_fd != -1) write_to_pipe(write_fd, outputs, 2);

        // wait for backward values
        if (read_back_fd != -1) {
            read_from_pipe(read_back_fd, back, 2);
            cout << "Layer " << layer_index << " Backward: " << back[0] << " " << back[1] << endl;
            outfile << back[0] << " " << back[1] << endl;
            if (layer_index > 0 && write_back_fd != -1) write_to_pipe(write_back_fd, back, 2);
        }

        // Second forward pass uses back values as inputs
        double* outputs2 = new double[2];
        outputs2[0] = back[0];
        outputs2[1] = back[1];
        cout << "Layer " << layer_index << " Forward Pass 2 outputs: " << outputs2[0] << " " << outputs2[1] << endl;
        outfile << outputs2[0] << " " << outputs2[1] << endl;
        if (!is_output && write_fd != -1) write_to_pipe(write_fd, outputs2, 2);

        delete[] outputs;
        delete[] outputs2;
        // close fds
        if (write_fd != -1) close(write_fd);
        if (read_back_fd != -1) close(read_back_fd);
        if (write_back_fd != -1) close(write_back_fd);
        return;
    }

    // Non-input layers: read inputs from previous layer
    double* inputs = new double[prev_neurons];
    if (read_fd != -1) read_from_pipe(read_fd, inputs, prev_neurons);

    // load weights for this layer starting at offset_start
    int off = offset_start;
    double** weights = get_weights(prev_neurons, curr_neurons, all_data, off);
    double* outputs = compute_layer_outputs(curr_neurons, prev_neurons, inputs, weights);
    cout << "Layer " << layer_index << " Forward Pass 1 outputs:";
    for (int i = 0; i < curr_neurons; ++i) cout << " " << outputs[i];
    cout << endl;
    for (int i = 0; i < curr_neurons; ++i) outfile << outputs[i] << " ";
    outfile << endl;
    if (!is_output) {
        if (write_fd != -1) write_to_pipe(write_fd, outputs, curr_neurons);
    } else {
        double sum = 0;
        for (int i = 0; i < curr_neurons; ++i) sum += outputs[i];
        double fx1 = (sum * sum + sum + 1) / 2;
        double fx2 = (sum * sum - sum) / 2;
        cout << "fx1: " << fx1 << ", fx2: " << fx2 << endl;
        outfile << fx1 << " " << fx2 << endl;
        back[0] = fx1; back[1] = fx2;
        if (write_back_fd != -1) write_to_pipe(write_back_fd, back, 2);
    }
    delete[] inputs;
    // keep weights in memory until after second forward pass by not deleting yet
    if (!is_output) {
        // Read backward values once (written by the next layer or output layer),
        // print and propagate them backwards once. Reuse the same values for
        // the second forward pass to avoid blocking on a second read.
        if (read_back_fd != -1) read_from_pipe(read_back_fd, back, 2);
        cout << "Layer " << layer_index << " Backward: " << back[0] << " " << back[1] << endl;
        outfile << back[0] << " " << back[1] << endl;
        if (layer_index > 0 && write_back_fd != -1) write_to_pipe(write_back_fd, back, 2);
    }

    // Second forward pass: reuse weights already loaded and the same backward values
    // read above. Do not perform another blocking read here.

    double* inputs2 = new double[prev_neurons];
    if (layer_index == 0) {
        inputs2[0] = back[0]; inputs2[1] = back[1];
    } else {
        if (read_fd != -1) read_from_pipe(read_fd, inputs2, prev_neurons);
    }
    double* outputs2 = compute_layer_outputs(curr_neurons, prev_neurons, inputs2, weights);
    cout << "Layer " << layer_index << " Forward Pass 2 outputs:";
    for (int i = 0; i < curr_neurons; ++i) cout << " " << outputs2[i];
    cout << endl;
    for (int i = 0; i < curr_neurons; ++i) outfile << outputs2[i] << " ";
    outfile << endl;
    if (!is_output) {
        if (write_fd != -1) write_to_pipe(write_fd, outputs2, curr_neurons);
    } else {
        double sum2 = 0;
        for (int i = 0; i < curr_neurons; ++i) sum2 += outputs2[i];
        cout << "Second forward sum: " << sum2 << endl;
        outfile << "Second forward sum: " << sum2 << endl;
    }

    delete[] inputs2;
    delete[] outputs2;
    for (int j = 0; j < curr_neurons; ++j) delete[] weights[j];
    delete[] weights;
    delete[] outputs;
    if (write_fd != -1) close(write_fd);
    if (read_fd != -1) close(read_fd);
    if (write_back_fd != -1) close(write_back_fd);
    if (read_back_fd != -1) close(read_back_fd);
}

int main() {
    int num_hidden;
    int num_neurons;

    cout << "Enter number of hidden layers: ";
    cin >> num_hidden;

    cout << "Enter number of neurons per hidden/output layer: ";
    cin >> num_neurons;

    // Input layer has fixed 2 neurons

    int data_size;
    double* all_data = readDoublesFromFile("input.txt", data_size);
    if (!all_data || data_size < 2) {
        cerr << "Not enough data in input.txt\n";
        return 1;
    }
    // First 2 values are initial inputs, rest are weights
    int total_layers = num_hidden + 2; // layer indices 0..total_layers-1

    // Precompute per-layer offsets into all_data (weights start after first 2 values)
    vector<int> offset_for_layer(total_layers, -1);
    int offset = 2;
    for (int li = 0; li < total_layers; ++li) {
        offset_for_layer[li] = offset;
        int curr_neurons = (li == 0 ? 2 : num_neurons);
        int prev_neurons = (li == 0 ? 0 : (li == 1 ? 2 : num_neurons));
        int count = (li == 0 ? 0 : prev_neurons * curr_neurons);
        offset += count;
    }

    // Create pipes between adjacent layers
    int n_pipes = max(0, total_layers - 1);
    vector<array<int,2>> forward_pipes(n_pipes);
    vector<array<int,2>> backward_pipes(n_pipes);
    for (int k = 0; k < n_pipes; ++k) {
        if (pipe(forward_pipes[k].data()) == -1) { perror("pipe"); return 1; }
        if (pipe(backward_pipes[k].data()) == -1) { perror("pipe"); return 1; }
    }

    vector<pid_t> children;
    for (int li = 0; li < total_layers; ++li) {
        pid_t pid = fork();
        if (pid == -1) { perror("fork"); return 1; }
        if (pid == 0) {
            // child: determine fds for this layer
            int read_fd = (li == 0 ? -1 : forward_pipes[li-1][0]);
            int write_fd = (li == total_layers-1 ? -1 : forward_pipes[li][1]);
            int read_back_fd = (li == total_layers-1 ? -1 : backward_pipes[li][0]);
            int write_back_fd = (li == 0 ? -1 : backward_pipes[li-1][1]);
            // close unused pipe ends in this child
            for (int k = 0; k < n_pipes; ++k) {
                if (li-1 != k) close(forward_pipes[k][0]);
                if (li != k) close(forward_pipes[k][1]);
                if (li != k) close(backward_pipes[k][0]);
                if (li-1 != k) close(backward_pipes[k][1]);
            }
            int curr_neurons = (li == 0 ? 2 : num_neurons);
            int prev_neurons = (li == 0 ? 0 : (li == 1 ? 2 : num_neurons));
            bool is_output = (li == total_layers - 1);
            process_layer(li, prev_neurons, curr_neurons, is_output, all_data, read_fd, write_fd, read_back_fd, write_back_fd, offset_for_layer[li]);
            // child done
            delete[] all_data;
            exit(0);
        } else {
            children.push_back(pid);
        }
    }

    // parent: close all pipe ends and wait for children
    for (int k = 0; k < n_pipes; ++k) {
        close(forward_pipes[k][0]); close(forward_pipes[k][1]);
        close(backward_pipes[k][0]); close(backward_pipes[k][1]);
    }
    for (pid_t c : children) waitpid(c, NULL, 0);

    delete[] all_data;
    return 0;
}