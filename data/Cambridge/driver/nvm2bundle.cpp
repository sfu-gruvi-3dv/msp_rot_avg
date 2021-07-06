#include <cstring>
#include <string.h>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <string.h>

using namespace std;

#include "../pba/DataInterface.h"

#include "../pba/util.h"

int char2int(char *x)
{
    int res = 0;
    int i = 0;
    while (x[i])
        res = res * 10 + x[i++] - '0';
    return res;
}

int main(int argc, char *argv[])
{

    printf("./a.out <input_file_name> <output_file_name>");

    char *input_filename = argv[1];
    char *output_filename = argv[2];

    printf("Input File: %s\n", input_filename);
    printf("Ouput File: %s\n", output_filename);

    vector<CameraT> camera_data;
    vector<Point3D> point_data;
    vector<Point2D> measurements;
    vector<int> camidx, ptidx;

    vector<string> photo_names;
    vector<int> point_color;

    ifstream fin(input_filename);
    ofstream fout(output_filename);
    int inoutcnt = 0;
    while (LoadNVM(fin, camera_data, point_data, measurements, ptidx, camidx, photo_names, point_color))
    {
        SaveBundlerOut(output_filename, camera_data, 
        point_data, measurements,ptidx,camidx, photo_names,point_color);

        printf("Number of Photo Names: %d\n", photo_names.size());

        camera_data.clear();
        point_data.clear();
        measurements.clear();
        ptidx.clear();
        camidx.clear();
        photo_names.clear();
        point_color.clear();
    }
    printf("Done");
    fin.close();
    fout.close();

    return 0;
}