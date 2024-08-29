#include "stdafx.h"
#include<stdio.h>
#include<string.h>
#include<limits.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<float.h>

#include "hmm.h"

// ai's using Durbin's algorithm
void durbinAlgo(){
	long double E = R[0];
	long double alpha[13][13];
	int i = 1;

	while (i <= P) {
		double k;
		long double numerator = R[i];
		long double alphaR = 0.0;
		int j = 1;

		while (j <= (i - 1)) {
			alphaR += alpha[j][i - 1] * R[i - j];
			j++;
		}
		numerator -= alphaR;
		k = numerator / E;
		alpha[i][i] = k;

		j = 1;
		while (j <= (i - 1)) {
			alpha[j][i] = alpha[j][i - 1] - (k * alpha[i - j][i - 1]);

			if (i == P) {
				a[j] = alpha[j][i];
			}
			j++;
		}

		// Update energy
		E = (1 - k * k) * E;

		if (i == P) {
			a[i] = alpha[i][i];
		}

		i++;
	}
}

//Calculate minimun LPC Coefficients using AutoCorrelation
void autoCorrelation(int frame_no) {
    long double s[FS];
    int sample_start_index = frame_no * 80;

    // Hamming window
    int i = 0;
    while (i < FS) {
        long double wn = 0.54 - 0.46 * cos((2 * (22.0 / 7.0) * i) / (FS - 1));
        s[i] = wn * samples[i + sample_start_index];
        i++;
    }

    // R0 to R12
    i = 0;
    while (i <= P) {
        long double sum = 0.0;
        int y = 0;
        while (y <= FS - 1 - i) {
            sum += (s[y] * s[y + i]);
            y++;
        }
        R[i] = sum;
        i++;
    }
    durbinAlgo();
}

// to get Cepstral Coefficient
void cepstralTransformation() {
    C[0] = 2.0 * (log(R[0]) / log(2.0));
    int m = 1;

    while (m <= P) {
        C[m] = a[m];
        int k = 1;

        while (k < m) {
            C[m] += (k * C[k] * a[m - k]) / m;
            k++;
        }
        m++;
    }
}
void raisedSineWindow() {
    int m = 1;
    while (m <= P) {
        long double wm = (1 + (Q / 2) * sin(pie * m / Q));
        C[m] *= wm;
        m++;
    }
}

void print_codebook(int k) {
    printf("\nSize: %d:\n", k);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < 12; j++) {
            printf("%Lf  ", codebook[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void initialize_with_centroid(){
	long double centroid[12] = {0.0};
	int i = 0;
	while (i < LBG_M) {
		int j = 0;
		while (j < 12) {
			centroid[j] += X[i][j];
			j++;
		}
		i++;
	}
	i = 0;
	while (i < 12) {
		centroid[i] /= LBG_M;
		codebook[0][i] = centroid[i];
		i++;
	}
	print_codebook(1);
}

long double calculate_distance(long double x[12], long double y[12]){
	long double distance = 0.0;
	int i = 0;
	while (i < 12) {
		distance += (tokhuraWeight[i + 1] * (x[i] - y[i]) * (x[i] - y[i]));
		i++;
	}
	return distance;
}

void nearest_neighbour(int k){
	int i = 0;
	while (i < LBG_M) {
		long double nn = DBL_MAX;
		int cluster_index;
		int j = 0;
		while (j < k) {
			long double dxy = calculate_distance(X[i], codebook[j]);
			if (dxy <= nn) {
				cluster_index = j;
				nn = dxy;
			}
			j++;
		}
		cluster[i] = cluster_index;
		i++;
	}
}

void codevector_update(int k){
	long double centroid[K][12] = {0.0};
	int n[K] = {0};
	int i = 0;
	while (i < LBG_M) {
		int j = 0;
		while (j < 12) {
			centroid[cluster[i]][j] += X[i][j];
			j++;
		}
		n[cluster[i]]++;
		i++;
	}
	i = 0;
	while (i < k) {
		int j = 0;
		while (j < 12) {
			codebook[i][j] = centroid[i][j] / n[i];
			j++;
		}
		i++;
	}
}

long double calculate_distortion(){
	long double distortion=0.0;
	int i=0;
	while(i<LBG_M){
		distortion+=calculate_distance(X[i],codebook[cluster[i]]);
		i++;
	}
	distortion/=LBG_M;
	return distortion;
}

void KMeans(int k){
	FILE* fp = fopen("distortion.txt", "a");
	if (fp == NULL) {
		printf("error in Kmeans, opening file!\n");
		return;
	}
	int m = 0;
	long double prev_D = DBL_MAX, cur_D = DBL_MAX;
	do {
		nearest_neighbour(k);
		m++;
		codevector_update(k);
		prev_D = cur_D;
		cur_D = calculate_distortion();
		//printf("m=%d\t:\tDistortion:%Lf\n", m, cur_D);
		fprintf(fp, "%Lf\n", cur_D);
	} while ((prev_D - cur_D) > DELTA);

	printf("K-means, update... ");
	print_codebook(k);
	fclose(fp);
}

void LBG(){
	int k=1;
	initialize_with_centroid();
	while (k != K) {
		int i = 0;
		while (i < k) {
			int j = 0;
			while (j < 12) {
				long double Yi = codebook[i][j];
				codebook[i][j] = Yi - EPSILON;
				codebook[i + k][j] = Yi + EPSILON;
				j++;
			}
			i++;
		}
		k = k * 2;
		KMeans(k);
	}
}


//Normalize the data
void normalize_data(char file[100]){
	// Open input file
	FILE* input_file = fopen(file, "r");
	if (input_file == NULL) {
		printf("Error in Opening File!\n");
		return;
	}

	int amplitude = 0, average_amplitude = 0;
	int i = 0;
	int sample_count = 0;
	int min_amplitude = INT_MAX;
	int max_amplitude = INT_MIN;
	while (!feof(input_file)) {
		fscanf(input_file, "%d", &amplitude);
		average_amplitude += amplitude;
		min_amplitude = (amplitude < min_amplitude) ? amplitude : min_amplitude;
		max_amplitude = (amplitude > max_amplitude) ? amplitude : max_amplitude;
		sample_count++;
	}

	average_amplitude /= sample_count;
	T = (sample_count - FS) / 80 + 1;

	if (T > T_) {
		T = T_;
	}

	min_amplitude -= average_amplitude;
	max_amplitude -= average_amplitude;

	fseek(input_file, 0, SEEK_SET);

	while (!feof(input_file)) {
		fscanf(input_file, "%d", &amplitude);

		if (min_amplitude == max_amplitude) {
			amplitude = 0;
		} else {
			amplitude -= average_amplitude;
			amplitude = (amplitude * CLIP) / ((max_amplitude > min_amplitude) ? max_amplitude : (-1) * min_amplitude);
			// Store normalized data
			samples[i++] = amplitude;
		}
	}

	fclose(input_file);

}

void calculate_energy_of_frame(int frame_no) {
    int sample_start_index = frame_no * 80;
    energy[frame_no] = 0;
    int i = 0;
    while (i < FS) {
        energy[frame_no] += samples[i + sample_start_index] * samples[i + sample_start_index];
        energy[frame_no] /= FS;
        i++;
    }
}

long double calculate_max_energy() {
    int nf = T;
    long double max_energy = DBL_MIN;
    int f = 0;
    while (f < nf) {
        if (energy[f] > max_energy) {
            max_energy = energy[f];
        }
        f++;
    }
    return max_energy;
}

long double calculate_avg_energy(){
	int nf=T;
	long double avg_energy=0.0;
	int f=0;
	while(f<nf){
		avg_energy+=energy[f];
		f++;
	}
	return avg_energy/nf;
}

//mark starting and ending of speech activity
void mark_checkpoints(){
	int nf = T;
    int f = 0;
    while (f < nf) {
        calculate_energy_of_frame(f);
        f++;
    }
	long double threshold_energy = calculate_avg_energy() / 10;
    int isAboveThresholdStart = 1;
    int isAboveThresholdEnd = 1;
    start_frame = 0;
    end_frame = nf - 1;

	f = 0;
    while (f < nf - 5) {
        int i = 0;
        while (i < 5) {
            isAboveThresholdStart *= (energy[f + i] > threshold_energy);
            i++;
        }
        if (isAboveThresholdStart) {
            start_frame = ((f - 5) > 0) ? (f - 5) : 0;
            break;
        }
        isAboveThresholdStart = 1;
        f++;
    }
	f = nf - 1;
    while (f > 4) {
        int i = 0;
        while (i < 5) {
            isAboveThresholdEnd *= (energy[f - i] > threshold_energy);
            i++;
        }
        if (isAboveThresholdEnd) {
            end_frame = ((f + 5) < nf) ? (f + 5) : (nf - 1);
            break;
        }
        isAboveThresholdEnd = 1;
        f--;
    }
}

void load_codebook(){
	FILE* fp;
	fp=fopen("234101002_codebook.csv","r");
	if(fp==NULL){
		printf("error in opening codebook file!\n");
		return;
	}
	int i = 1;
	while (i <= M) {
		int j = 1;
		while (j <= Q) {
			fscanf(fp, "%Lf,", &reference[i][j]);
			j++;
		}
		i++;
	}
	fclose(fp);
}
//store coefficients
void process_universe_file(FILE* fp, char file[]){
	normalize_data(file);
	int m=0;
	int nf=T;
	int f=0;
	while(f<nf){
		autoCorrelation(f);
		cepstralTransformation();
		raisedSineWindow();
		for(int i=1;i<=Q;i++){
			fprintf(fp,"%Lf,",C[i]);
		}
		fprintf(fp,"\n");
		f++;
	}
}

void generate_universe(){
	FILE *universefp;
	universefp=fopen("234101002_universe.csv","w");
	int d = 0;
	while (d <= 9) {
		int u = 1;
		while (u <= TRAIN_SIZE) {
			char filename[40];
			_snprintf(filename, 40, "234101002_dataset/234101002_E_%d_%d.txt", d, u);
			process_universe_file(universefp, filename);
			u++;
		}
		d++;
	}
}


int minTokhuraDistance(long double testC[]){
	long double minimum_distance = DBL_MAX;
	int min_distance_index = 0;
	int i = 1;
	while (i <= M) {
		long double distance = 0.0;
		int j = 1;
		while (j <= Q) {
			distance += (tokhuraWeight[j] * (testC[j] - reference[i][j]) * (testC[j] - reference[i][j]));
			j++;
		}
		if (distance < minimum_distance) {
			minimum_distance = distance;
			min_distance_index = i;
		}
		i++;
	}
	return min_distance_index;
}

void generate_observation_sequence(char file[]){
	FILE* fp = fopen("o.txt", "w");
	normalize_data(file);
	int m = 0;
	mark_checkpoints();
	T = (end_frame - start_frame + 1);
	int nf = T;
	int f = start_frame;

	while (f <= end_frame) {
		autoCorrelation(f);
		cepstralTransformation();
		raisedSineWindow();
		fprintf(fp, "%d ", minTokhuraDistance(C));
		f++;
	}
	fprintf(fp, "\n");
	fclose(fp);
}

void load_universe(char file[100]){
	FILE* fp=fopen(file,"r");
	if(fp==NULL){
		printf("Error in opening universe file!\n");
		return;
	}
	int i=0;
	long double c;
	while(!feof(fp)){
		fscanf(fp,"%Lf,",&c);
		X[LBG_M][i]=c;
		i=(i+1)%12;
		if(i==0) LBG_M++;
	}
	fclose(fp);
}

void store_codebook(char file[100],int k){
	FILE* fp=fopen(file,"w");
	if(fp==NULL){
		printf("Error in file for store_codebook()\n");
		return;
	}
	int i = 0;
	while (i < k) {
		int j = 0;
		while (j < 12) {
			fprintf(fp, "%Lf,", codebook[i][j]);
			j++;
		}
		fprintf(fp, "\n");
		i++;
	}
	fclose(fp);
}

void generate_codebook(){
	load_universe("234101002_universe.csv");
	LBG();
	store_codebook("234101002_codebook.csv",K);
}

void set_initial_model(){
	int d=0;
	while(d<=9){
		char srcfnameA[40];
		_snprintf(srcfnameA,40,"initial/A_%d.txt",d);
		char srcfnameB[40];
		_snprintf(srcfnameB,40,"initial/B_%d.txt",d);
		char destfnameA[40];
		_snprintf(destfnameA,40,"initial_model/A_%d.txt",d);
		char destfnameB[40];
		_snprintf(destfnameB,40,"initial_model/B_%d.txt",d);
		char copyA[100];
		_snprintf(copyA,100,"copy /Y %s %s",srcfnameA,destfnameA);
		char copyB[100];
		_snprintf(copyB,100,"copy /Y %s %s",srcfnameB,destfnameB);
		system(copyA);
		system(copyB);
		d++;
	}
	
}

void initial_model(int d){
	FILE *fp;
	initialization();
	char filenameA[40];
	_snprintf(filenameA,40,"initial_model/A_%d.txt",d);
	fp = fopen(filenameA, "r");
	if (fp == NULL)
		printf("Error in opening A\n");
	int i = 1;
	while (i <= N) {
		int j = 1;
		while (j <= N) {
			fscanf(fp, "%Lf ", &A[i][j]);
			j++;
		}
		i++;
	}
	fclose(fp);
	char filenameB[40];
	_snprintf(filenameB,40,"initial_model/B_%d.txt",d);
	fp = fopen(filenameB, "r");
	i = 1;
	while (i <= N) {
		int j = 1;
		while (j <= M) {
			fscanf(fp, "%Lf ", &B[i][j]);
			j++;
		}
		i++;
	}
	fclose(fp);
	fp = fopen("initial_model/pi.txt", "r");
	i=1;
	while(i<=N){
		fscanf(fp, "%Lf ", &pi[i]);
		i++;
	}
	fclose(fp);
	fp=fopen("o.txt","r");
	i=1;
	while(i <= T){
		fscanf(fp, "%d\t", &O[i]);
		i++;
	}
	fclose(fp);
}

void train_model(int digit, int utterance){
	int m=0;
	do{
		alphaCalc();
		calculate_beta();
		calculate_gamma();
		P_star_dash=P_star;
		viterbi_algo();
		calculate_xi();
		re_estimation();
		m++;
		printf("Iteration: %d\n",m);
	}while(m<60 && P_star > P_star_dash);
	printf("Digit: %d \nP* value = %e\n",digit,P_star);
	FILE *fp;
	char filenameA[40];
	_snprintf(filenameA,40,"234101002_lambda/A_%d_%d.txt",digit,utterance);
	fp=fopen(filenameA,"w");
	int i = 1;
	while (i <= N) {
		int j = 1;
		while (j <= N) {
			fprintf(fp, "%e ", A[i][j]);
			j++;
		}
		fprintf(fp, "\n");
		i++;
	}
	fclose(fp);
	char filenameB[40];
	_snprintf(filenameB,40,"234101002_lambda/B_%d_%d.txt",digit,utterance);
	fp=fopen(filenameB,"w");
	i = 1;
	while (i <= N) {
		int j = 1;
		while (j <= M) {
			fprintf(fp, "%e ", B[i][j]);
			j++;
		}
		fprintf(fp, "\n");
		i++;
	}
	fclose(fp);
}

void calculate_avg_model_param(int d){
	long double A_sum[N+1][N+1] = {0}, B_sum[N+1][M+1] = {0}, temp;
	FILE* fp;
	int u = 1;
	while (u <= 25) {
		char filenameA[40];
		_snprintf(filenameA, 40, "234101002_lambda/A_%d_%d.txt", d, u);
		fp = fopen(filenameA, "r");
		int i = 1;
		while (i <= N) {
			int j = 1;
			while (j <= N) {
				fscanf(fp, "%Lf ", &temp);
				A_sum[i][j] += temp;
				j++;
			}
			i++;
		}
		fclose(fp);
		char filenameB[40];
		_snprintf(filenameB, 40, "234101002_lambda/B_%d_%d.txt", d, u);
		fp = fopen(filenameB, "r");
		i = 1;
		while (i <= N) {
			int j = 1;
			while (j <= M) {
				fscanf(fp, "%Lf ", &temp);
				B_sum[i][j] += temp;
				j++;
			}
			i++;
		}
		fclose(fp);
		u++;
	}
	FILE* avgfp;
	char fnameA[40];
	_snprintf(fnameA,40,"initial_model/A_%d.txt",d);
	avgfp=fopen(fnameA,"w");
	int i = 1;
	while (i <= N) {
		int j = 1;
		while (j <= N) {
			A[i][j] = A_sum[i][j] / 25;
			fprintf(avgfp, "%e ", A[i][j]);
			j++;
		}
		fprintf(avgfp, "\n");
		i++;
	}

	fclose(avgfp);
	char fnameB[40];
	_snprintf(fnameB,40,"initial_model/B_%d.txt",d);
	avgfp=fopen(fnameB,"w");
	i = 1;
	while (i <= N) {
		int j = 1;
		while (j <= M) {
			B[i][j] = B_sum[i][j] / 25;
			fprintf(avgfp, "%e ", B[i][j]);
			j++;
		}
		fprintf(avgfp, "\n");
		i++;
	}

	fclose(avgfp);
}

void store_final_lambda(int digit){
	FILE *fp;
	char filenameA[40];
	_snprintf(filenameA,40,"234101002_lambda/A_%d.txt",digit);
	fp=fopen(filenameA,"w");
	int i = 1;
	while (i <= N) {
		int j = 1;
		while (j <= N) {
			fprintf(fp, "%e ", A[i][j]);
			j++;
		}
		fprintf(fp, "\n");
		i++;
	}
	fclose(fp);
	char filenameB[40];
	_snprintf(filenameB,40,"234101002_lambda/B_%d.txt",digit);
	fp=fopen(filenameB,"w");
	i = 1;
	while (i <= N) {
		int j = 1;
		while (j <= M) {
			fprintf(fp, "%e ", B[i][j]);
			j++;
		}
		fprintf(fp, "\n");
		i++;
	}
	fclose(fp);
}

void processTestFile(int d){
	FILE *fp;
	initialization();
	char filenameA[40];
	_snprintf(filenameA,40,"234101002_lambda/A_%d.txt",d);
	fp=fopen(filenameA,"r");
	if (fp == NULL)
		printf("Error in processTestFile()\n");
	int i = 1;
	while (i <= N) {
		int j = 1;
		while (j <= N) {
			fscanf(fp, "%Lf ", &A[i][j]);
			j++;
		}
		i++;
	}
	fclose(fp);

	char filenameB[40];
	_snprintf(filenameB,40,"234101002_lambda/B_%d.txt",d);
	fp=fopen(filenameB,"r");
	i = 1;
	while (i <= N) {
		int j = 1;
		while (j <= M) {
			fscanf(fp, "%Lf ", &B[i][j]);
			j++;
		}
		i++;
	}
	fclose(fp);
	fp = fopen("initial_model/pi.txt", "r");
	i=1;
	while(i<=N){
		fscanf(fp, "%Lf ", &pi[i]);
		i++;
	}
	fclose(fp);

	fp=fopen("o.txt","r");
	i=1;
	while(i<=T){
		fscanf(fp, "%d\t ", &O[i]);
		i++;
	}
	fclose(fp);
}
