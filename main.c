#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define inLayerCount 17
#define hidLayerCount 9
#define outLayerCount 10
#define m 2216
#define alpha 0.05
#define epsn 0.01

int calculateNorm1(int w,int h,double a[][w]){

	int i,j;
	double ss=0;
	for(i=0;i<h;i++){
		for(j=0;j<w;j++){
			ss+=a[i][j]*a[i][j];
		}
	}
	ss=sqrt(ss);
	// printf("%lf\n",ss);
	if(ss<epsn)
		return 1;
	return 0;
}

void LoadData(int a[][inLayerCount]){
	FILE *fp;
	fp=fopen("train1.txt","r");
	if(fp==NULL){
		printf("Cannot open Test file...\n");
		return;
	}
	int i,x=0,y=0;
	fscanf(fp,"%d",&i);
	a[y][x]=i;
	while(!feof(fp)){
		fscanf(fp,"%d",&i);
		x++;
		x=x%17;
		if(x==0)y++;
		a[y][x]=i;
	}
	fclose(fp);
}

void TestData(int a[][inLayerCount]){
	FILE *fp;
	fp=fopen("test.txt","r");
	if(fp==NULL){
		printf("Cannot open Test file...\n");
		return;
	}
	int i,x=0,y=0;
	fscanf(fp,"%d",&i);
	a[y][x]=i;
	while(!feof(fp)){
		fscanf(fp,"%d",&i);
		x++;
		x=x%17;
		if(x==0)y++;
		a[y][x]=i;
	}
	fclose(fp);
}

void MakeRandomWeightMatrix(int w,int h,double wt[][w],int d){
	int i,j;
	for(i=0;i<h;i++){
		for(j=0;j<w;j++){
		 	// wt[i][j]=(double)((unsigned)rand()+1)/(double)((unsigned)RAND_MAX+2);
		 	if(d==1)
				wt[i][j] = (rand()%8 +1)/100.0;
			else
				wt[i][j] = (rand()%10 +1)/10.0 - 0.5;
	 }
	}
}

void getData(int inMat[],int data[3000][inLayerCount],double outMat_y[],int p){
    int i;
	inMat[0]=1;
	for(i=1;i<inLayerCount;i++){
		inMat[i]=data[p][i];
	}
	for(i=0;i<outLayerCount;i++){
		outMat_y[i]=0;
	}
	outMat_y[data[p][0]-1]=1;
}

double sigmoid(double x){
     double exp_value;
     double return_value;

     /*** Exponential calculation ***/
     exp_value = exp(-x);
     // printf("exp::::%lf %lf ",exp_value,x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);
     // printf("return_value::::%lf\n",return_value);
     return return_value;
}

void multiply1(int inMat[],double wt1[][inLayerCount],double hidMat[]){
	int i,j;
	double val=0;
	hidMat[0]=1;
	for(i=0;i<hidLayerCount-1;i++){
		val=0;
		for(j=0;j<inLayerCount;j++){
			val+=wt1[i][j]*inMat[j];
		}
		hidMat[i+1]=sigmoid(val);
	}
}

void multiply2(double hidMat[],double wt[][hidLayerCount],double outMat[]){
	double val=0;

	int i,j;
	for(i=0;i<outLayerCount;i++){
		val=0;
		for(j=0;j<hidLayerCount;j++){
			val+=wt[i][j]*hidMat[j];
		}
		outMat[i]=sigmoid(val);
	}
}

int sumOfSquaredLoss(double outMat_y[], double outMat[], double hidMat[], int inMat[], double wt1[][inLayerCount], double wt2[][hidLayerCount],int x){
	double error_out[outLayerCount];
	double del_out[outLayerCount][hidLayerCount]={0};
	double del_hid[hidLayerCount-1][inLayerCount]={0};

	int i,j,k,l;
	double sum=0;
	for(i=0;i<outLayerCount;i++){
		error_out[i]=outMat[i]-outMat_y[i];
	}

	for(i=0;i<outLayerCount;i++){
		error_out[i]=error_out[i]*outMat[i]*(1-outMat[i]);
	}

	for(i=0;i<outLayerCount;i++){
		for(j=0;j<hidLayerCount;j++){
			del_out[i][j]=error_out[i]*hidMat[j];
		}
	}

	for(l=0;l<hidLayerCount-1;l++){
		for(i=0;i<inLayerCount;i++){
			for(j=0;j<outLayerCount;j++){
				sum=sum+error_out[j]*wt2[j][l];
			}
			del_hid[l][i]=hidMat[l+1]*(1-hidMat[l+1])*sum*inMat[i];
			sum=0;
		}
	}

	for(i=0;i<outLayerCount;i++){
		for(j=0;j<hidLayerCount;j++){
			wt2[i][j]-=del_out[i][j]*alpha;
		}
	}

	for(i=0;i<hidLayerCount-1;i++){
		for(j=0;j<inLayerCount;j++){
			wt1[i][j]-=del_hid[i][j]*alpha;
		}
	}

	if(x==1 && calculateNorm1(inLayerCount,hidLayerCount-1,del_hid) && calculateNorm1(outLayerCount,hidLayerCount,del_out))
		return 1;
	else return 0;
	
}

void crossEntropy(double outMat_y[], double outMat[],double hidMat[], double wt2[][hidLayerCount],int inMat[],double Del_out[][hidLayerCount],double Del_hid[][inLayerCount]){
	double del_out[outLayerCount];
	double del_hid[hidLayerCount-1];
	double res[hidLayerCount-1]={0};

    int i,j;

	for(i=0;i<outLayerCount;i++){
		del_out[i]=outMat[i]-outMat_y[i];
	}

	for(i=1;i<hidLayerCount;i++){
		for(j=0;j<outLayerCount;j++){
			del_hid[i-1]+=wt2[j][i]*del_out[j];
		}
		del_hid[i-1]*=hidMat[i]*(1-hidMat[i]);
	}

	for(i=0;i<outLayerCount;i++){
		for(j=0;j<hidLayerCount;j++){
			Del_out[i][j]+=(hidMat[j]*del_out[i]*alpha)/m;
		}
	}

	for(i=0;i<hidLayerCount-1;i++){
		for(j=0;j<inLayerCount;j++){
			Del_hid[i][j]+=(inMat[j]*del_hid[i]*alpha)/m;
		}
	}

}

int main(){

    int i,j,k,l;
	int inMat[inLayerCount]={0};
	double hidMat[hidLayerCount]={0};
	double outMat[outLayerCount]={0};
	double outMat_y[outLayerCount]={0};
	int data[3000][17],data1[1000][17],d,x=0,c=0,counter=100;

	printf("Loading Data...\n");
	LoadData(data);
	printf("Loading Data Done...\n");
	printf("--------------------------\n");

	double wt1[hidLayerCount-1][inLayerCount];
	double wt2[outLayerCount][hidLayerCount];

	

	printf("Enter 1 for sumOfSquaredLoss and 2 for crossEntropy\n");
	scanf("%d", &d);
	printf("Enter 1 for delta stopping criterion and 2 for epoche criterion\n");
	scanf("%d", &x);

	MakeRandomWeightMatrix(inLayerCount,hidLayerCount-1,wt1,d);
	MakeRandomWeightMatrix(outLayerCount,hidLayerCount,wt2,d);

	// for(i=0;i<hidLayerCount-1;i++){
	// 	for(j=0;j<inLayerCount;j++){
	// 		printf("%lf ",wt1[i][j]);
	// 	}
	// 	printf("\n");
	// }
	// 	printf("-----------------------------------------\n");

	// for(i=0;i<outLayerCount;i++){
	// 	for(j=0;j<hidLayerCount;j++){
	// 		printf("%lf ",wt2[i][j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("-----------------------------------------\n");

	while(1){
		double Del_out[outLayerCount][hidLayerCount]={0};
		double Del_hid[hidLayerCount-1][inLayerCount]={0};
		for(k=0;k<m;k++){
			getData(inMat,data,outMat_y,k);
			multiply1(inMat,wt1,hidMat);
			multiply2(hidMat,wt2,outMat);
			if(d==1){
				c=sumOfSquaredLoss(outMat_y,outMat,hidMat,inMat,wt1,wt2,x);
			}
			else{
				crossEntropy(outMat_y,outMat,hidMat,wt2,inMat,Del_out,Del_hid);
			}
		}

		if(x==1 && c==1)
				break;


		if(d==2){
			for(i=0;i<hidLayerCount-1;i++){
				for(j=0;j<inLayerCount;j++){
					wt1[i][j]-=Del_hid[i][j];
				}
			}

			for(i=0;i<outLayerCount;i++){
				for(j=0;j<hidLayerCount;j++){
					wt2[i][j]-=Del_out[i][j];
				}
			}
			if(x==1 && calculateNorm1(hidLayerCount-1,outLayerCount,Del_hid) && calculateNorm1(outLayerCount,hidLayerCount,Del_out)){
				break;
			}
		}

		if(x==2){
			// int count=0;
			// for(i=0;i<998;i++){
			// 	TestData(data1);
			// 	getData(inMat,data1,outMat_y,i);
			// 	multiply1(inMat,wt1,hidMat);
			// 	multiply2(hidMat,wt2,outMat);
			// 	int j,maxout=0;
			// 	for(j=0;j<outLayerCount;j++){
			// 		if(outMat[maxout]<outMat[j])
			// 			maxout=j;
			// 	}
			// 	if(maxout+1==data1[i][0]){
			// 		count++;
			// 	}
			// }
			// printf("%d %.2lf\n",101-counter,(count/998.0)*100);

			counter--;
			if(counter==0)
				break;
		}

	}

	// for(i=0;i<hidLayerCount-1;i++){
	// 	for(j=0;j<inLayerCount;j++){
	// 		printf("%lf ",wt1[i][j]);
	// 	}
	// 	printf("\n");
	// }
	// 	printf("-----------------------------------------\n");

	// for(i=0;i<outLayerCount;i++){
	// 	for(j=0;j<hidLayerCount;j++){
	// 		printf("%lf ",wt2[i][j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("-----------------------------------------\n");

	int count=0;
	for(i=0;i<998;i++){

		TestData(data1);
		getData(inMat,data1,outMat_y,i);
		multiply1(inMat,wt1,hidMat);
		multiply2(hidMat,wt2,outMat);
		int j,maxout=0;
		for(j=0;j<outLayerCount;j++){
			if(outMat[maxout]<outMat[j])
				maxout=j;
		}
		if(maxout+1==data1[i][0]){
			count++;
		}
	}
	printf("No of neurons used : %d\n",hidLayerCount-1);
	printf("Accuracy:%.2lf\n",(count/998.0)*100);

	return 0;
}