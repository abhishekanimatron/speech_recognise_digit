#include "stdafx.h"
#include<stdio.h>
#include<string.h>
#include<limits.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<float.h>


#define K 32					//LBG Codebook Size
#define DELTA 0.00001			//K-Means Parameter
#define EPSILON 0.03			 //LBG Splitting Parameter
#define UNIVERSE_SIZE 50000		//Universe Size
#define CLIP 5000				//Max value after normalizing
#define FS 320					//Frame Size
#define Q 12					//No. of cepstral coefficient
#define P 12					//No. of LPC
#define pie (22.0/7)
#define N 5						//no. of states in HMM Model
#define M 32					//Codebook Size
#define T_ 400					//Max possible no. of frames
#define TRAIN_SIZE 20			//Training Files for each utterance
#define TEST_SIZE 50			//Total Test Files if Train Size is 25

//HMM Model Variables
long double A[N + 1][N + 1];
long double B[N + 1][M + 1];
long double pi[N + 1];
long double alpha[T_ + 1][N + 1];
long double beta[T_ + 1][N + 1];
long double gamma[T_ + 1][N + 1];
long double delta[T_+1][N+1];
long double xi[T_+1][N+1][N+1];
long double A_bar[N + 1][N + 1];
long double B_bar[N + 1][M + 1];
long double pi_bar[N + 1];

int O[T_+1];
int q[T_+1];
int psi[T_+1][N+1];
int q_star[T_+1];

long double P_star = -1;
long double P_star_dash = -1;

int samples[50000];
int T=160;
int start_frame;
int end_frame;

long double R[P+1];
long double a[P+1];
long double C[Q+1];
long double reference[M+1][Q+1];
long double tokhuraWeight[Q+1]={0.0, 1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};
long double energy[T_]={0};
long double X[UNIVERSE_SIZE][Q];
int LBG_M=0;
long double codebook[K][Q];
int cluster[UNIVERSE_SIZE];


//Initialize every variable of HMM module to zero
void initialization() {
    int i=1, j=1;
    for (; i <= N; i++) {
        for (j = 1; j <= N; j++) {
            A[i][j] = 0;
        }
        for (j = 1; j <= M; j++) {
            B[i][O[j]] = 0;
        }
        pi[i] = 0;
    }
    int t = 1;
    while (t <= T) {
        int j = 1;
        while (j <= N) {
            alpha[t][j] = 0;
            beta[t][j] = 0;
            gamma[t][j] = 0;
            j++;
        }
        t++;
    }
}
// forward procedure, alpha calculation
void alphaCalc()
{
    int i, j;
    // initialize alpha values for the first observation
    for (i = 1; i <= N; i++)
    {
        alpha[1][i] = pi[i] * B[i][O[1]];
    }
    int t = 1;
    while (t < T)
    {
        j = 1;
        while (j <= N)
        {
            long double sum = 0;
            int i = 1;
            while (i <= N)
            {
                // calculate the sum for alpha[t+1][j]
                sum += alpha[t][i] * A[i][j];
                i++;
            }
            // update alpha values for the next time step
            alpha[t + 1][j] = sum * B[j][O[t + 1]];
            j++;
        }
        t++;
    }
    // Write the alpha values to a file
    FILE *fp = fopen("alpha.txt", "w");
    t = 1;
    while (t <= T)
    {
        int j = 1;
        while (j <= N)
        {
            fprintf(fp, "%e\t", alpha[t][j]);
            j++;
        }
        fprintf(fp, "\n");
        t++;
    }
    fclose(fp);
}

// evaluating model, 1st problem's solution
long double calculate_score() {
    long double probability = 0;
    int i = 1;
    while (i <= N) {
        probability += alpha[T][i];
        i++;
    }
    return probability;
}

// backward procedure, beta calculation
void calculate_beta()
{
    int i, j;

    // initialize beta values for the last time step
    for (i = 1; i <= N; i++)
    {
        beta[T][i] = 1;
    }
    int t = T - 1;
    while (t >= 1)
    {
        i = 1;
        while (i <= N)
        {
            j = 1;
            while (j <= N)
            {
                // calculate the sum for beta[t][i]
                beta[t][i] += A[i][j] * B[j][O[t + 1]] * beta[t + 1][j];
                j++;
            }
            i++;
        }
        t--;
    }
    // write the beta values to a file
    FILE *fp = fopen("beta.txt", "w");
    t = 1;
    while (t < T)
    {
        int j = 1;
        while (j <= N)
        {
            fprintf(fp, "%e\t", beta[t][j]);
            j++;
        }
        fprintf(fp, "\n");
        t++;
    }
    fclose(fp);
}

// to predict the most likely state sequence using gamma values
void predict_state_sequence()
{
    int t = 1;
    while (t <= T)
    {
        long double max = 0;
        int index = 0;
        int j = 1;
        while (j <= N)
        {
            // state with the highest gamma value
            if (gamma[t][j] > max)
            {
                max = gamma[t][j];
                index = j;
            }
            j++;
        }
        // store the most likely state at time t
        q[t] = index;
        t++;
    }
    // write observed sequence and predicted sequence
    FILE *fp = fopen("predicted_seq_gamma.txt", "w");
    t = 1;
    while (t <= T)
    {
        fprintf(fp, "%4d\t", O[t]);
        t++;
    }
    fprintf(fp, "\n");
    t = 1;
    while (t <= T)
    {
        fprintf(fp, "%4d\t", q[t]);
        t++;
    }
    fprintf(fp, "\n");
    fclose(fp);
}
// to calculate gamma values
void calculate_gamma()
{
    int t = 1;
    while (t <= T)
    {
        long double sum = 0;
        int i = 1;
        while (i <= N)
        {
            sum += alpha[t][i] * beta[t][i];
            i++;
        }
        i = 1;
        while (i <= N)
        {
            gamma[t][i] = alpha[t][i] * beta[t][i] / sum;
            i++;
        }
        t++;
    }
    // write it to file
    FILE *fp = fopen("gamma.txt", "w");
    t = 1;
    while (t <= T)
    {
        int j = 1;
        while (j <= N)
        {
            fprintf(fp, "%.16e\t", gamma[t][j]);
            j++;
        }
        fprintf(fp, "\n");
        t++;
    }
    fclose(fp);
    predict_state_sequence();
}

// 2nd problem's solution:
void viterbi_algo()
{
    int i, j, t;
    // initialization step for the Viterbi
    i = 1;
    for (; i <= N; i++)
    {
        // delta initial
        delta[1][i] = pi[i] * B[i][O[1]];
        psi[1][i] = 0; // backtrack from initial
    }
    t = 2;
    // recursion step for the Viterbi algorithm
    while (t <= T)
    {
        j = 1;
        while (j <= N)
        {
            long double max = DBL_MIN;
            int index = 0;
            i = 1;
            while (i <= N)
            {
                // calculate the maximum delta and its state
                if (delta[t - 1][i] * A[i][j] > max)
                {
                    max = delta[t - 1][i] * A[i][j];
                    index = i;
                }
                i++;
            }
            delta[t][j] = max * B[j][O[t]]; // update delta values
            psi[t][j] = index;              // store the backtracking path
            j++;
        }
        t++;
    }
    // termination step
    P_star = DBL_MIN;
    i = 1;
    while (i <= N)
    {
        if (delta[T][i] > P_star)
        { // for the highest probability
            P_star = delta[T][i];
            q_star[T] = i; // most likely state at time T
        }
        i++;
    }
    // find the most likely state sequence
    t = T - 1;
    while (t >= 1)
    {
        // backtrack through psi matrix
        q_star[t] = psi[t + 1][q_star[t + 1]];
        t--;
    }

    // writing observed sequence and most likely state sequence
    FILE *fp = fopen("predicted_seq_viterbi.txt", "w");
    t = 1;
    while (t <= T)
    {
        fprintf(fp, "%4d\t", O[t]);
        t++;
    }
    fprintf(fp, "\n");
    t = 1;
    while (t <= T)
    {
        fprintf(fp, "%4d\t", q_star[t]);
        t++;
    }
    fprintf(fp, "\n");
    fclose(fp);
}

// xi values
void calculate_xi() {
    int t = 1;
    while (t < T) {
        long double denominator = 0.0;
        int i, j;
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                denominator += (alpha[t][i] * A[i][j] * B[j][O[t + 1]] * beta[t + 1][j]);
            }
        }
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                xi[t][i][j] = (alpha[t][i] * A[i][j] * B[j][O[t + 1]] * beta[t + 1][j]) / denominator;
            }
        }
        t++;
    }
}

// 3rd problem's solution
// for re-estimation of HMM parameters
void re_estimation()
{
    int i, j, k, t;
    // Initial state probabilities
    i = 1;
    while (i <= N)
    {
        pi_bar[i] = gamma[1][i];
        i++;
    }

    // Update state transition prob.
    i = 1;
    while (i <= N)
    {
        int mi = 0;
        long double maxValue = DBL_MIN;
        long double adjustSum = 0;

        // Most probable transition & adjust the prob.
        j = 1;
        while (j <= N)
        {
            long double numerator = 0.0, denominator = 0.0;
            t = 1;
            while (t <= T - 1)
            {
                // Calculate transition prob. based on gamma & xi values
                numerator += xi[t][i][j];
                denominator += gamma[t][i];
                t++;
            }
            A_bar[i][j] = (numerator / denominator);
            if (A_bar[i][j] > maxValue)
            {
                maxValue = A_bar[i][j];
                mi = j;
            }
            adjustSum += A_bar[i][j];
            j++;
        }

        // Adjust transition prob.
        A_bar[i][mi] += (1 - adjustSum);
        i++;
    }

    // update emission prob.
    j = 1;
    while (j <= N)
    {
        int mi = 0;
        long double maxValue = DBL_MIN;
        long double adjustSum = 0;

        // Most probable emission and adjust the probabilities
        int k = 1;
        while (k <= M)
        {
            long double numerator = 0.0, denominator = 0.0;
            int t = 1;
            while (t <= T)
            {
                if (O[t] == k)
                {
                    numerator += gamma[t][j];
                }
                denominator += gamma[t][j];
                t++;
            }
            B_bar[j][k] = (numerator / denominator);
            if (B_bar[j][k] > maxValue)
            {
                maxValue = B_bar[j][k];
                mi = k;
            }
            if (B_bar[j][k] < 1.00e-030)
            {
                B_bar[j][k] = 1.00e-030;
            }
            adjustSum += B_bar[j][k];
            k++;
        }
        B_bar[j][mi] += (1 - adjustSum);
        j++;
    }

    i = 1;
    while (i <= N)
    {
        pi[i] = pi_bar[i];
        i++;
    }

    i = 1;
    while (i <= N)
    {
        int j = 1;
        while (j <= N)
        {
            A[i][j] = A_bar[i][j];
            j++;
        }
        i++;
    }

    j = 1;
    while (j <= N)
    {
        int k = 1;
        while (k <= M)
        {
            B[j][k] = B_bar[j][k];
            k++;
        }
        j++;
    }
}