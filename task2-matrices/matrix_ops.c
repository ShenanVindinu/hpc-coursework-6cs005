/*
    Task 2 - matrix operations, parallelised with OpenMP.

    Reads a file containing a sequence of matrices (two numbers "rows,cols" on
    their own line, then that many comma-separated rows), treats them as
    consecutive pairs, and for each pair tries: add, subtract, elementwise
    multiply, elementwise divide, transpose of each matrix, and a full matrix
    product. Whatever doesn't fit the shapes involved gets a short message
    instead of a crash. Results for pair N land in results_pairN.txt.

    usage: ./matrix_ops <data file> <threads>
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef struct {
    int rows, cols;
    double *values; // flattened, row-major
} Grid;

#define ELEM(g, r, c) ((g)->values[(r) * (g)->cols + (c)])

/* ---------- loading the input file ---------- */

static int next_dims(FILE *fp, int *r, int *c) {
    char buf[256];
    do {
        if (!fgets(buf, sizeof(buf), fp)) return 0; // nothing left to read
    } while (buf[0] == '\n' || buf[0] == '\r');

    if (sscanf(buf, "%d,%d", r, c) != 2 || *r <= 0 || *c <= 0) {
        fprintf(stderr, "bad dimension line: '%s'\n", buf);
        return -1;
    }
    return 1;
}

static Grid *read_grid_values(FILE *fp, int r, int c) {
    Grid *g = malloc(sizeof(Grid));
    g->rows = r;
    g->cols = c;
    g->values = malloc(sizeof(double) * r * c);

    char buf[4096];
    for (int row = 0; row < r; row++) {
        if (!fgets(buf, sizeof(buf), fp)) {
            fprintf(stderr, "file cut off early - wanted %d rows, only got %d\n", r, row);
            free(g->values); free(g);
            return NULL;
        }
        char *piece = strtok(buf, ",\r\n");
        for (int col = 0; col < c; col++) {
            if (!piece) {
                fprintf(stderr, "row %d is missing values (wanted %d)\n", row, c);
                free(g->values); free(g);
                return NULL;
            }
            char *end;
            double val = strtod(piece, &end);
            if (end == piece) {
                fprintf(stderr, "'%s' on row %d isn't a number\n", piece, row);
                free(g->values); free(g);
                return NULL;
            }
            ELEM(g, row, col) = val;
            piece = strtok(NULL, ",\r\n");
        }
    }
    return g;
}

static Grid **read_all_grids(const char *path, int *total) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "can't open '%s'\n", path);
        exit(1);
    }

    int cap = 8, n = 0;
    Grid **arr = malloc(sizeof(Grid *) * cap);

    for (;;) {
        int r, c;
        int status = next_dims(fp, &r, &c);
        if (status == 0) break;
        if (status < 0) { fclose(fp); exit(1); }

        Grid *g = read_grid_values(fp, r, c);
        if (!g) { fclose(fp); exit(1); }

        if (n == cap) { cap *= 2; arr = realloc(arr, sizeof(Grid *) * cap); }
        arr[n++] = g;
    }

    fclose(fp);
    *total = n;
    return arr;
}

static Grid *new_grid(int r, int c) {
    Grid *g = malloc(sizeof(Grid));
    g->rows = r;
    g->cols = c;
    g->values = malloc(sizeof(double) * r * c);
    return g;
}

static void drop_grid(Grid *g) {
    if (!g) return;
    free(g->values);
    free(g);
}

// don't bother spinning up more threads than there are output rows to split
// between them
static int thread_budget(int wanted, int rows_available) {
    return wanted < rows_available ? wanted : rows_available;
}

/* ---------- the seven operations ---------- */

static Grid *do_add(Grid *a, Grid *b, int wanted) {
    Grid *out = new_grid(a->rows, a->cols);
    int t = thread_budget(wanted, out->rows);
    #pragma omp parallel for num_threads(t)
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            ELEM(out, i, j) = ELEM(a, i, j) + ELEM(b, i, j);
    return out;
}

static Grid *do_sub(Grid *a, Grid *b, int wanted) {
    Grid *out = new_grid(a->rows, a->cols);
    int t = thread_budget(wanted, out->rows);
    #pragma omp parallel for num_threads(t)
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            ELEM(out, i, j) = ELEM(a, i, j) - ELEM(b, i, j);
    return out;
}

static Grid *do_elem_mul(Grid *a, Grid *b, int wanted) {
    Grid *out = new_grid(a->rows, a->cols);
    int t = thread_budget(wanted, out->rows);
    #pragma omp parallel for num_threads(t)
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            ELEM(out, i, j) = ELEM(a, i, j) * ELEM(b, i, j);
    return out;
}

static Grid *do_elem_div(Grid *a, Grid *b, int wanted) {
    Grid *out = new_grid(a->rows, a->cols);
    int t = thread_budget(wanted, out->rows);
    #pragma omp parallel for num_threads(t)
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++) {
            double d = ELEM(b, i, j);
            ELEM(out, i, j) = (d == 0.0) ? NAN : ELEM(a, i, j) / d;
        }
    return out;
}

static Grid *do_transpose(Grid *a, int wanted) {
    Grid *out = new_grid(a->cols, a->rows);
    int t = thread_budget(wanted, out->rows);
    #pragma omp parallel for num_threads(t)
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            ELEM(out, j, i) = ELEM(a, i, j);
    return out;
}

static Grid *do_matmul(Grid *a, Grid *b, int wanted) {
    Grid *out = new_grid(a->rows, b->cols);
    int t = thread_budget(wanted, out->rows);
    #pragma omp parallel for num_threads(t)
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double acc = 0.0;
            for (int k = 0; k < a->cols; k++) {
                acc += ELEM(a, i, k) * ELEM(b, k, j);
            }
            ELEM(out, i, j) = acc;
        }
    }
    return out;
}

/* ---------- writing results ---------- */

static void dump_grid(FILE *out, Grid *g) {
    for (int i = 0; i < g->rows; i++) {
        for (int j = 0; j < g->cols; j++) {
            double v = ELEM(g, i, j);
            if (isnan(v)) fprintf(out, "NaN");
            else fprintf(out, "%.4f", v);
            if (j < g->cols - 1) fprintf(out, ",");
        }
        fprintf(out, "\n");
    }
}

static void handle_pair(int idx, Grid *a, Grid *b, int wanted) {
    char fname[64];
    snprintf(fname, sizeof(fname), "results_pair%d.txt", idx);
    FILE *out = fopen(fname, "w");
    if (!out) {
        fprintf(stderr, "couldn't write '%s'\n", fname);
        return;
    }

    int equal_shape = (a->rows == b->rows && a->cols == b->cols);
    int chainable = (a->cols == b->rows);

    fprintf(out, "Pair %d: A(%d,%d) B(%d,%d)\n\n", idx, a->rows, a->cols, b->rows, b->cols);

    if (equal_shape) {
        Grid *r1 = do_add(a, b, wanted);
        fprintf(out, "Addition - (%d,%d)\n", r1->rows, r1->cols);
        dump_grid(out, r1); fprintf(out, "\n");
        drop_grid(r1);

        Grid *r2 = do_sub(a, b, wanted);
        fprintf(out, "Subtraction - (%d,%d)\n", r2->rows, r2->cols);
        dump_grid(out, r2); fprintf(out, "\n");
        drop_grid(r2);

        Grid *r3 = do_elem_mul(a, b, wanted);
        fprintf(out, "Element-wise Multiplication - (%d,%d)\n", r3->rows, r3->cols);
        dump_grid(out, r3); fprintf(out, "\n");
        drop_grid(r3);

        Grid *r4 = do_elem_div(a, b, wanted);
        fprintf(out, "Element-wise Division - (%d,%d)\n", r4->rows, r4->cols);
        dump_grid(out, r4); fprintf(out, "\n");
        drop_grid(r4);
    } else {
        fprintf(out, "Addition cannot be done (shapes differ).\n\n");
        fprintf(out, "Subtraction cannot be done (shapes differ).\n\n");
        fprintf(out, "Element-wise Multiplication cannot be done (shapes differ).\n\n");
        fprintf(out, "Element-wise Division cannot be done (shapes differ).\n\n");
    }

    Grid *ta = do_transpose(a, wanted);
    fprintf(out, "Transpose of A - (%d,%d)\n", ta->rows, ta->cols);
    dump_grid(out, ta); fprintf(out, "\n");
    drop_grid(ta);

    Grid *tb = do_transpose(b, wanted);
    fprintf(out, "Transpose of B - (%d,%d)\n", tb->rows, tb->cols);
    dump_grid(out, tb); fprintf(out, "\n");
    drop_grid(tb);

    if (chainable) {
        Grid *prod = do_matmul(a, b, wanted);
        fprintf(out, "Matrix Multiplication (A x B) - (%d,%d)\n", prod->rows, prod->cols);
        dump_grid(out, prod);
        drop_grid(prod);
    } else {
        fprintf(out, "Matrix Multiplication cannot be done (A.cols != B.rows).\n");
    }

    fclose(out);
    printf("pair %2d: A(%2dx%2d) B(%2dx%2d) -> %s\n", idx, a->rows, a->cols, b->rows, b->cols, fname);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <data file> <threads>\n", argv[0]);
        return 1;
    }

    int wanted_threads = atoi(argv[2]);
    if (wanted_threads < 1) {
        fprintf(stderr, "thread count needs to be at least 1\n");
        return 1;
    }

    int total;
    Grid **grids = read_all_grids(argv[1], &total);

    int pairs = total / 2;
    if (total % 2 != 0) {
        fprintf(stderr, "note: %d grids loaded - odd one out at the end was skipped\n", total);
    }

    for (int p = 0; p < pairs; p++) {
        handle_pair(p + 1, grids[p * 2], grids[p * 2 + 1], wanted_threads);
    }

    printf("\nfinished - %d pairs written to results_pair1.txt .. results_pair%d.txt\n", pairs, pairs);

    for (int i = 0; i < total; i++) drop_grid(grids[i]);
    free(grids);
    return 0;
}
