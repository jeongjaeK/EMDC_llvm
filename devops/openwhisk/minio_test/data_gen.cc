#include <fstream>
#include <string>
#include <iostream>
#include <bits/stdc++.h>


int main(int argc, char* argv[]){
    if(argc != 3){
        std::cout << "How to use ...\n  \
                    ./exe [file_name] [size]\n";
    }

    FILE *file = fopen(argv[1],"wb");

    size_t len = std::stoi(argv[2]);

    fwrite(&len, sizeof(size_t), 1, file);

    float *arr;
    size_t size = sizeof(float) * len;
    arr = (float*)malloc(size);

    for(auto j=1; j<=3; ++j){
        for(auto i=0; i<len; ++i){
            arr[i] = i*j / 16.0f;
        }
        fwrite(arr, sizeof(float), size, file);
    }

    fclose(file);


    return 0;
}