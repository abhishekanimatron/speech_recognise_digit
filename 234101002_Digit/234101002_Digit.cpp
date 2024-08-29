// 234101002_Digit.cpp : Defines the entry point for the console application
#include "stdafx.h"
#include<stdio.h>
#include<string.h>
#include<limits.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<float.h>
#include<Windows.h>
#include<iostream>
#include<fstream>
using namespace std;

#include "preprocess.h"

void displayThings(){
	cout << "CS566 Speech Processing: Assignment of Digit Recognition\n";
    cout << "Roll no. 234101002\n";
	cout<<"--------------------------------------------------------\n";
	cout << "Processing...\n\n";
}

int recognize_digit(){
	int rec_digit=10;
	long double max_prob=DBL_MIN;
	int d=0;
	while(d<=9){
		processTestFile(d);
		alphaCalc();
		long double prob=calculate_score();
		printf("P(O|lambda%d)=%e\n",d,prob);
		if(prob>max_prob){
			max_prob=prob;
			rec_digit=d;
		}
		d++;
	}
	return rec_digit;
}

void train_HMM(){
	set_initial_model();
	int d = 0;
	while (d <= 9) {
		int t = 1;
		while (t <= 2) {
			int u = 1;
			while (u <= TRAIN_SIZE) {
				char filename[40];
				_snprintf(filename, 40, "234101002_dataset/234101002_E_%d_%d.txt", d, u);
				generate_observation_sequence(filename);
				initial_model(d);
				train_model(d, u);
				u++;
			}
			calculate_avg_model_param(d);
			t++;
		}
		store_final_lambda(d);
		d++;
	}
}

void test_HMM(){
	double accuracy=0.0;
	int d = 0;
	while (d <= 9) {
		int u = 21;
		while (u <= 30) {
			char filename[40];
			_snprintf(filename, 40, "234101002_dataset/234101002_E_%d_%d.txt", d, u);
			generate_observation_sequence(filename);
			printf("Current : %d", d);
			int rd = recognize_digit();
			printf("\tPredicted: %d\n", rd);
			if (rd == d) {
				accuracy += 1.0;
			}
			u++;
		}
		d++;
	}
	printf("\nAchieved accuracy of %f\n",accuracy);
}

void process_live_data(char filename[100]){
	FILE *fp;
	char prefixf[100]="live_input/";
	strcat(prefixf,filename);
	fp=fopen(prefixf,"r");
	int samples[13000];
	int x=0;
	for(int i=0;!feof(fp);i++){
		fscanf(fp,"%d",&x);
		if(i>=6000 && i<19000)
			samples[i-6000]=x;
	}
	fclose(fp);
	char prefix[100]="live_input/processed_";
	strcat(prefix,filename);
	fp=fopen(prefix,"w");
	int i=0;
	while(i<13000){
		fprintf(fp,"%d\n",samples[i]);
		i++;
	}
	fclose(fp);
}

void live_test_HMM(){
	Sleep(2000);
	system("Recording_Module.exe 2 live_input/test.wav live_input/test.txt");
	generate_observation_sequence("live_input/test.txt");
	int rd=recognize_digit();
	printf("Predicted digit: %d\n",rd);		
}
int _tmain(int argc, _TCHAR* argv[])
{
	displayThings();
	int menu_choice;
	cout<<"-------------MENU---------------\n";
	cout<<"1.Training & testing on files\n";
	cout<<"2.Live testing\n";
	cout<<"--------------------------------\n";
	cout<<"Enter your choice: \n";
	cin>>menu_choice;
	cout<<"--------------------------------\n";
	switch(menu_choice){
		case 1:
			generate_universe();
			generate_codebook();
			load_codebook();
			train_HMM();
			test_HMM();
			break;
		case 2:
			load_codebook();
			printf("Press enter to stop recording!\n");
			live_test_HMM();
			break;
		default:cout<<"Invalid choice. Terminating.\n";
	}
	getchar();
	return 0;
}


