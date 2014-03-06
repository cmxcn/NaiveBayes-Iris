/******************************************************************************************
 *���������ڶ�iris�����������ر�Ҷ˹����                                                  *
 *��vc++2010�����±���������гɹ�                                                        *
 *���ߣ�������                                                                            *
 ******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#define CNUM 3
#define ANUM 4
#define PI   3.141592653
double Pac[CNUM][ANUM]; // �洢Cj����������ai�����ĸ���
double Pc[CNUM];   // �洢Ci�������������
double mean[CNUM][ANUM];   //�洢��i�������µ�j�����Եľ�ֵ
double stde[CNUM][ANUM];   //�洢��i�������µ�j�����Եı�׼��


/*�����˹�����ܶȺ���ֵ*/
double GaussPos_Calc(double x, double mean, double stde)
{
    double ret = 0;
    ret = exp(-(x-mean)*(x-mean)/(2*stde*stde)) / (sqrt(2*PI)*stde);
    return ret;
}

/*�����ֵ�ͱ�׼��ĺ���
 *�������ݣ�����ָ�룬���ݸ������洢��ֵָ�룬 �洢��׼��ָ��
 *������ݣ���ֵ�ͱ�׼��
 *����ֵ����
 */
void Mean_Stde_Calc(double *data, int num, double *mean, double *stde)
{
    int i;
    double sum = 0, sqre = 0;  // �ͺͷ���

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

/*�ҳ�����������Ԫ��
 *���룺����ָ�룬���鳤��
 *��������Ԫ�ص��±�
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


/*����ѧϰ��������ѧϰ�л�ȡ��������*/
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
		//��ȡ��������
        fscanf(datafile, "%lf,%lf,%lf,%lf,Iris-%s", tempdata, tempdata+1, tempdata+2, tempdata+3, irisName);
		//��ȡ���з�
        fgetc(datafile);
		//�ж�ʵ�ʷ���
        for( i = 0; i < 3; i++ )
            if( strcmp(irisClass[i], irisName) == 0 )
                break;
		//��i����������+1
        index[i]++;
		//��¼��Ӧ��������ĸ�����������ֵ
        for( j = 0; j < ANUM; j++ )
            attr[i][j][index[i]] = tempdata[j];

    }

	//���ݶ�ȡ��ϣ���ʼ��������
	//��Ϊ����������ֵ��ʵ��������Ҫ���������������ÿ�����Եķ���;�ֵ�����ո�˹�ֲ����������
    for( i = 0; i < CNUM; i++ )
        for( j = 0; j < ANUM; j++ )
    {
		//�����i�����������j�����Եķ���;�ֵ
        Mean_Stde_Calc(attr[i][j], index[i]+1, &mean[i][j], &stde[i][j]);
    }
	//����ÿ��������ֵ��������
    for( i = 0; i < CNUM; i++ )
      Pc[i] = ((double)(index[i]+1))/(index[0]+ index[1] + index[2] + 3);
}

/*У�����������Է���Ч��*/
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
		//��ȡ��������
        fscanf(datafile, "%lf,%lf,%lf,%lf,Iris-%s", tempdata, tempdata+1, tempdata+2, tempdata+3, irisName);
		//��ȡ���з�
        fgetc(datafile);

		//ͳ���ܵ��������������ڼ�����ȷ��
        wholeNum++;
		//������x���������·���ci�������������ʣ�ʵ��ֻ����x��ciͬʱ�����ĸ��ʣ�����ĵ�˵����
        for( i = 0; i < CNUM; i++ )
        {
            pcx[i] = Pc[i];
            for( j = 0; j < ANUM; j++ )
                pcx[i] *= GaussPos_Calc(tempdata[j], mean[i][j], stde[i][j]);
        }

		//ȡ��x������ci�������������Ǹ�������Ϊ������
        ans = Max_Ele(pcx, CNUM);


        printf("����%d:��ȷ���ࣺiris-%s; �������:iris-%s", wholeNum, irisName, irisClass[ans]);
        if(!strcmp(irisName,irisClass[ans]) )
        {
            rightNum++;//�ȽϽ�����洢��ȷ��������
			printf(" ������ȷ\n");
        }
		else printf(" �������\n");
    }

    rightRate = ((double)rightNum)/wholeNum;
    printf("\n����������%d����ȷ��������%d ��ȷ�ʣ�%lf\n", wholeNum, rightNum, rightRate);

}

int main()
{
    FILE *teachFile = fopen("irislearn.txt", "r");
    FILE *checkFile = fopen("irischeck.txt", "r");

	if( teachFile == NULL )
	{
		printf("�Ҳ���ѧϰ�����ļ��������������Ŀ¼��\n");
	}
	else if( checkFile == NULL )
	{
		printf("�Ҳ������������ļ��������������Ŀ¼��\n");
	}
	else
	{
		learn(teachFile); //һ��������ѧϰѵ������
		check(checkFile); //��һ����������ʵ��Ч��
	}
	getchar();
	fclose(teachFile);
	fclose(checkFile);
    return 0;
}
