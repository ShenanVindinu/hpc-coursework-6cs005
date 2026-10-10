#include <stdio.h>
#include <stdlib.h>
#include <time.h>

char* cudaCrypt(char* rawPassword) {
    static char newPassword[11];

    newPassword[0] = rawPassword[0] + 2;
    newPassword[1] = rawPassword[0] - 2;
    newPassword[2] = rawPassword[0] + 1;
    newPassword[3] = rawPassword[1] + 3;
    newPassword[4] = rawPassword[1] - 3;
    newPassword[5] = rawPassword[1] - 1;
    newPassword[6] = rawPassword[2] + 2;
    newPassword[7] = rawPassword[2] - 2;
    newPassword[8] = rawPassword[3] + 4;
    newPassword[9] = rawPassword[3] - 4;
    newPassword[10] = '\0';

    for(int i = 0; i < 10; i++) {
        if(i >= 0 && i < 6) {
            if(newPassword[i] > 122) {
                newPassword[i] = (newPassword[i] - 122) + 97;
            } else if(newPassword[i] < 97) {
                newPassword[i] = (97 - newPassword[i]) + 97;
            }
        } else {
            if(newPassword[i] > 57) {
                newPassword[i] = (newPassword[i] - 57) + 48;
            } else if(newPassword[i] < 48) {
                newPassword[i] = (48 - newPassword[i]) + 48;
            }
        }
    }
    return newPassword;
}

int main(void) {
    int n;

    printf("Enter number of random passwords to generate: ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("Invalid number.\n");
        return 1;
    }

    FILE *fp = fopen("passwords.txt", "w");
    if (fp == NULL) {
        perror("Error opening output file");
        return 1;
    }

    srand((unsigned int) time(NULL));

    char rawPassword[5];
    rawPassword[4] = '\0';

    for (int i = 0; i < n; i++) {
        rawPassword[0] = 'a' + (rand() % 26);
        rawPassword[1] = 'a' + (rand() % 26);
        rawPassword[2] = '0' + (rand() % 10);
        rawPassword[3] = '0' + (rand() % 10);

        char *encrypted = cudaCrypt(rawPassword);

        fprintf(fp, "%s\n", encrypted);
    }

    fclose(fp);

    printf("Generated %d passwords and saved them to passwords.txt\n", n);

    return 0;
}
