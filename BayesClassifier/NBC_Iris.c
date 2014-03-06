/******************************************************************************************
 *本程序用于对iris样本进行朴素贝叶斯分类                                                  *
 *在vc++2010环境下编译调试运行成功                                                        *
 *作者：陈梦翔                                                                            *
 ******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#define CNUM 3
#define ANUM 4
#define PI   3.141592653
double Pac[CNUM][ANUM]; // 存储Cj发生条件下ai发生的概率
double Pc[CNUM];   // 存储Ci发生的先验概率
double mean[CNUM][ANUM];   //存储第i个分类下第j个属性的均值
double stde[CNUM][ANUM];   //存储第i个分类下第j个属性的标准差


/*计算高斯概率密度函数值*/
double GaussPos_Calc(double x, double mean, double stde)
{
    double ret = 0;
    ret = exp(-(x-mean)*(x-mean)/(2*stde*stde)) / (sqrt(2*PI)*stde);
    return ret;
}

/*计算均值和标准差的函数
 *输入数据：数据指针，数据个数，存储均值指针， 存储标准差指针
 *输出数据：均值和标准差
 *返回值：无
 */
void Mean_Stde_Calc(double *data, int num, double *mean, double *stde)
{
    int i;
    double sum = 0, sqre = 0;  // 和和方差

    for( i = 0; i < num; i++ )
        sum += data[i];

    *mean = sum/num;
    for( i = 0; i < num; i++ )
    {
        sqre += (data[i] - *mean)*(data[i] - *mean);
    }
    sqre /= num;
    *stde = sqrt(sqre);
}

/*找出数组中最大的元素
 *输入：数组指针，数组长度
 *输出：最大元素的下标
*/
int Max_Ele(double *data, int num)
{
    double tempMax = data[0];
    int maxindex = 0;
    int i;
    for( i = 1; i < num; i++ )
        if( data[i] > tempMax )
        {

            tempMax = data[i];
            maxindex = i;
        }
    return maxindex;
}


/*样本学习函数，从学习中获取条件概率*/
void learn(FILE *datafile)
{
    double attr[CNUM][ANUM][100] = {{{0}}};
    double tempdata[ANUM] = {0};
    char irisClass[3][15] = {"setosa", "versicolor", "virginica"};

    char irisName[20] = {0};
    int index[3] = {-1, -1, -1};
    int i,j;
    while( !feof(datafile) )
    {
		//读取输入数据
        fscanf(datafile, "%lf,%lf,%lf,%lf,Iris-%s", tempdata, tempdata+1, tempdata+2, tempdata+3, irisName);
		//读取换行符
        fgetc(datafile);
		//判断实际分类
        for( i = 0; i < 3; i++ )
            if( strcmp(irisClass[i], irisName) == 0 )
                break;
		//第i个分类数量+1
        index[i]++;
		//记录对应分类下面的各个属性数据值
        for( j = 0; j < ANUM; j++ )
            attr[i][j][index[i]] = tempdata[j];

    }

	//数据读取完毕，开始处理数据
	//因为属性是连续值（实数），需要计算出各个分类下每个属性的方差和均值，按照高斯分布来计算概率
    for( i = 0; i < CNUM; i++ )
        for( j = 0; j < ANUM; j++ )
    {
		//计算第i个分类下面第j个属性的方差和均值
        Mean_Stde_Calc(attr[i][j], index[i]+1, &mean[i][j], &stde[i][j]);
    }
	//计算每个分类出现的先验概率
    for( i = 0; i < CNUM; i++ )
      Pc[i] = ((double)(index[i]+1))/(index[0]+ index[1] + index[2] + 3);
}

/*校验样本，测试分类效果*/
void check(FILE *datafile)
{

    double tempdata[ANUM] = {0};
    int rightNum = 0;
    int wholeNum = 0;
    double rightRate = 0;
    char irisClass[3][15] = {"setosa", "versicolor", "virginica"};
    double pcx[CNUM] = {0};
    char irisName[20] = {0};
    int i,j,ans;
    while( !feof(datafile) )
    {
		//读取输入数据
        fscanf(datafile, "%lf,%lf,%lf,%lf,Iris-%s", tempdata, tempdata+1, tempdata+2, tempdata+3, irisName);
		//读取换行符
        fgetc(datafile);

		//统计总的样本数量，用于计算正确率
        wholeNum++;
		//计算在x发生条件下分类ci发生的条件概率（实际只求了x和ci同时发生的概率，详见文档说明）
        for( i = 0; i < CNUM; i++ )
        {
            pcx[i] = Pc[i];
            for( j = 0; j < ANUM; j++ )
                pcx[i] *= GaussPos_Calc(tempdata[j], mean[i][j], stde[i][j]);
        }

		//取出x条件下ci发生概率最大的那个分类作为分类结果
        ans = Max_Ele(pcx, CNUM);


        printf("样本%d:正确分类：iris-%s; 程序分类:iris-%s", wholeNum, irisName, irisClass[ans]);
        if(!strcmp(irisName,irisClass[ans]) )
        {
            rightNum++;//比较结果，存储正确分类数量
			printf(" 分类正确\n");
        }
		else printf(" 分类错误\n");
    }

    rightRate = ((double)rightNum)/wholeNum;
    printf("\n测试总数：%d，正确分类数：%d 正确率：%lf\n", wholeNum, rightNum, rightRate);

}

int main()
{
    FILE *teachFile = fopen("irislearn.txt", "r");
    FILE *checkFile = fopen("irischeck.txt", "r");

	if( teachFile == NULL )
	{
		printf("找不到学习样本文件！请放在主程序目录下\n");
	}
	else if( checkFile == NULL )
	{
		printf("找不到测试样本文件！请放在主程序目录下\n");
	}
	else
	{
		learn(teachFile); //一半样本的学习训练过程
		check(checkFile); //另一半样本检验实际效果
	}
	getchar();
	fclose(teachFile);
	fclose(checkFile);
    return 0;
}
