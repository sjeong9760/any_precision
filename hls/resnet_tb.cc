#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

int readfile(float* test, char* filename); 
bool check(int n, float* test, float* target);

int main(void){

    bool correct = true;
    //Layer 1
    float* layer1 = (float *) malloc(sizeof(float) * 262144);
    float* layer1_hls = (float *) malloc(sizeof(float) * 262144);
    int param_num = readfile(layer1, "layer1.txt");

    // call hls function
    // resnet(layer1_hls);    

    // check hls function is correct
    if (!check(param_num, layer1, layer1_hls)){
        correct = false;
    }

    if (correct) printf("Pass!");
    else printf("Fail!");


    return 0;
}

int readfile(float* test, char* filename){
    int i = 0;
    float value;
    const int max = 50;
    char line[max];
    FILE *fp = fopen(filename, "r");
    while(!feof(fp)) {
        fgets(line,max,fp);
        value = atof(line);
        test[i] = value;
        i++;
    }

    return i;
}

bool check(int n, float* test, float* target){
    for(int i=0; i<n; i++){
        if (test[i] - target[i] > 0.001){
            return false;
        }
    }
    return true;
}