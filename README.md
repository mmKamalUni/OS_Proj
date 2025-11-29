# Multi-Core Neural Network Simulation (OS Project)

Build

```bash
g++ -o main main.cpp -pthread -Wall -Wextra
```

Run

```bash
./main
```

The program will prompt:
- Enter number of hidden layers: (integer)
- Enter number of neurons per hidden/output layer: (integer)

Input file

Place `input.txt` in the same directory. The file should contain comma- or whitespace-separated doubles. The first two numbers are the input layer values; the rest are weights in sequence (layer-wise). Example lines from sample `Instructions/input.txt` are comma-separated; the parser accepts commas.

Output

Results (forward/backward values and second forward sums) are appended to `output.txt`.

Notes

- The program uses processes for layers and POSIX threads for neurons. Pipes implement IPC. Mutexes protect shared outputs within a layer.
- This is a demo/educational implementation matching the course instructions.
