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

    printf("./a.out <input_file_name> <output_file_name> <co_vis_lower_bound> ");
    printf("<co_vis_upper_bound> <driver_argument>\n");

    char *input_filename = argv[1];
    char *output_filename = argv[2];
    int covis_lower_bound = char2int(argv[3]);
    int covis_upper_bound = char2int(argv[4]);
    char *driver_argument = argv[argc - 1];

    printf("Input File: %s\n", input_filename);
    printf("Ouput File: %s\n", output_filename);
    printf("covis_lower_bound: %d\n", covis_lower_bound);
    printf("covis_upper_bound: %d\n", covis_upper_bound);

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
        vector<vector<int>> cnt(point_data.size());
        vector<vector<int>> inoutmat;
        for(int i=0;i<camera_data.size();i++)
        {
            inoutmat.push_back(vector<int>(camera_data.size(),0));
        }

        printf("cnt size: %d\n", cnt.size());
        printf("inoutmat size: %d, %d\n", inoutmat.size(), inoutmat[0].size());
        for (int i = 0; i < ptidx.size(); i++)
        {
            cnt[ptidx[i]].push_back(camidx[i]);
        }
        printf("pt cam link done\n");
        for (int i = 0; i < cnt.size(); i++)
        {
            vector<int> vec = cnt[i];
            //printf("i=%d ,vec[i] size=%d, cnt size=%d\n",i,vec.size(),cnt.size());
            for (int j = 0; j < vec.size(); j++)
            {
                //printf("j = %d\n",j);
                for (int k = 0; k < j; k++)
                {
                    //printf("k = %d\n",k);
                    int x = vec[j];
                    int y = vec[k];
                    //printf("x = %d, y = %d\n",x,y);
                    if (x > y)
                        swap(x, y);
                    inoutmat[x][y]++;
                    //if(inoutmat[x][y]>1000)printf("%d %d inoutmat %d\n",x,y,inoutmat[x][y]);
                }
            }
        }
        printf("Inoutmat done.\n");
        printf("%d %d %d\n", 50,51,inoutmat[50][51]);
        for (int i = 0; i < camera_data.size(); i++)
        {
            for (int j = i+1; j < camera_data.size(); j++)
            {
                printf("%d %d %d\n", i, j, inoutmat[i][j]);
                if (inoutmat[i][j] >= covis_lower_bound && inoutmat[i][j] <= covis_upper_bound)
                {
                    inoutcnt++;
                    printf("%s %s\n", photo_names[i].c_str(), photo_names[j].c_str());
                    fout << photo_names[i].c_str() << " " << photo_names[j].c_str()<<endl;
                }
            }
        }
        // Clear all data for next model
        camera_data.clear();
        point_data.clear();
        measurements.clear();
        ptidx.clear();
        camidx.clear();
        photo_names.clear();
        point_color.clear();
    }
    printf("Output done. %d edges\n", inoutcnt);

    fin.close();
    fout.close();

    return 0;
}