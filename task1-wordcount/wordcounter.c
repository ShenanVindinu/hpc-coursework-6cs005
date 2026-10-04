/*
 * 6CS005 - Task 1
 * This Counts how often each word shows up in a text file, splitting the work
 * across a number of threads given on the command line. Threads work on
 * their own slice of the file, then combine results into one shared list
 * (protected by a mutex so nothing gets corrupted while multiple threads
 * try to update it).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>

#define CAP 10000       // max distinct words we can store
#define NAME_LIMIT 64    // longest word we'll keep (truncated past this)

struct Entry {
    char text[NAME_LIMIT];
    int hits;
};

struct Dictionary {
    struct Entry list[CAP];
    int count;
};

// the final combined results live here, shared across all threads
static struct Dictionary global_dict;
static pthread_mutex_t dict_lock = PTHREAD_MUTEX_INITIALIZER;

struct Segment {
    long from;
    long to;
};

struct WorkerData {
    int id;
    const char *text;
    long from;
    long to;
};

static void dict_reset(struct Dictionary *d) {
    d->count = 0;
}

// bumps the count for a word if it's already there, otherwise adds it.
// caller is responsible for making sure this isn't called from two threads
// on the same dictionary at once.
static void dict_insert(struct Dictionary *d, const char *w) {
    int i;
    for (i = 0; i < d->count; i++) {
        if (strcmp(d->list[i].text, w) == 0) {
            d->list[i].hits += 1;
            return;
        }
    }

    if (d->count >= CAP) {
        return; // silently drop - CAP is generous enough that this shouldn't trigger
    }

    strncpy(d->list[d->count].text, w, NAME_LIMIT - 1);
    d->list[d->count].text[NAME_LIMIT - 1] = '\0';
    d->list[d->count].hits = 1;
    d->count++;
}

// slurps the whole file into a malloc'd buffer and null-terminates it
static char *load_whole_file(const char *path, long *size_out) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "couldn't open '%s' - check the path\n", path);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);

    char *data = malloc(len + 1);
    if (data == NULL) {
        fprintf(stderr, "ran out of memory loading the file\n");
        fclose(f);
        exit(1);
    }

    fread(data, 1, len, f);
    data[len] = '\0';
    fclose(f);

    *size_out = len;
    return data;
}

// splits [0, size) into n roughly-equal pieces, nudging each boundary
// forward to land on whitespace so we never cut a word in two
static struct Segment *split_into_segments(const char *text, long size, int n) {
    struct Segment *segs = malloc(sizeof(struct Segment) * n);
    long share = size / n;
    long cursor = 0;

    for (int k = 0; k < n; k++) {
        segs[k].from = cursor;

        if (k == n - 1) {
            segs[k].to = size; // last piece mops up any leftover bytes
        } else {
            long edge = cursor + share;
            while (edge < size && text[edge] != ' ' && text[edge] != '\n' && text[edge] != '\t') {
                edge++;
            }
            segs[k].to = edge;
        }

        cursor = segs[k].to;
    }

    return segs;
}

// this is what each thread actually runs - tokenize its own slice into a
// private dictionary first (no locking needed for that part), then merge
// into the shared one with the lock held only for the merge step
static void *thread_job(void *raw) {
    struct WorkerData *job = (struct WorkerData *)raw;

    struct Dictionary mine;
    dict_reset(&mine);

    char buf[NAME_LIMIT];
    int len = 0;

    for (long pos = job->from; pos <= job->to; pos++) {
        char ch = (pos < job->to) ? job->text[pos] : ' '; // pretend there's a space at the end to flush the last word

        if (isalnum((unsigned char)ch)) {
            if (len < NAME_LIMIT - 1) {
                buf[len++] = (char)tolower((unsigned char)ch);
            }
        } else if (len > 0) {
            buf[len] = '\0';
            dict_insert(&mine, buf);
            len = 0;
        }
    }

    printf("[thread %d] scanned bytes %ld-%ld, found %d unique words in its slice\n",
           job->id, job->from, job->to, mine.count);

    pthread_mutex_lock(&dict_lock);
    for (int i = 0; i < mine.count; i++) {
        int matched = 0;
        for (int j = 0; j < global_dict.count; j++) {
            if (strcmp(global_dict.list[j].text, mine.list[i].text) == 0) {
                global_dict.list[j].hits += mine.list[i].hits;
                matched = 1;
                break;
            }
        }
        if (!matched && global_dict.count < CAP) {
            global_dict.list[global_dict.count] = mine.list[i];
            global_dict.count++;
        }
    }
    pthread_mutex_unlock(&dict_lock);

    return NULL;
}

// qsort comparator - highest hit count first
static int by_hits_desc(const void *a, const void *b) {
    const struct Entry *ea = a;
    const struct Entry *eb = b;
    return eb->hits - ea->hits;
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s <input file> <thread count> [output file]\n", argv[0]);
        return 1;
    }

    const char *in_path = argv[1];
    int n_threads = atoi(argv[2]);
    const char *out_path = (argc == 4) ? argv[3] : "result.txt";

    if (n_threads < 1) {
        fprintf(stderr, "thread count has to be at least 1\n");
        return 1;
    }

    long file_size;
    char *text = load_whole_file(in_path, &file_size);
    struct Segment *segs = split_into_segments(text, file_size, n_threads);

    dict_reset(&global_dict);

    pthread_t *tids = malloc(sizeof(pthread_t) * n_threads);
    struct WorkerData *jobs = malloc(sizeof(struct WorkerData) * n_threads);

    for (int i = 0; i < n_threads; i++) {
        jobs[i].id = i;
        jobs[i].text = text;
        jobs[i].from = segs[i].from;
        jobs[i].to = segs[i].to;
        pthread_create(&tids[i], NULL, thread_job, &jobs[i]);
    }

    for (int i = 0; i < n_threads; i++) {
        pthread_join(tids[i], NULL);
    }

    qsort(global_dict.list, global_dict.count, sizeof(struct Entry), by_hits_desc);

    FILE *out = fopen(out_path, "w");
    if (out == NULL) {
        fprintf(stderr, "couldn't write to '%s'\n", out_path);
        free(tids); free(jobs); free(segs); free(text);
        return 1;
    }

    for (int i = 0; i < global_dict.count; i++) {
        fprintf(out, "%s %d\n", global_dict.list[i].text, global_dict.list[i].hits);
    }
    fclose(out);

    printf("\nfinished - %d unique words across %d threads, written to %s (sorted high to low)\n",
           global_dict.count, n_threads, out_path);

    free(tids);
    free(jobs);
    free(segs);
    free(text);
    pthread_mutex_destroy(&dict_lock);
    return 0;
}
