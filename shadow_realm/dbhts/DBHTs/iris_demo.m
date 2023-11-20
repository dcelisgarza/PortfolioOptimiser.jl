% This script performs clustering on Fisher's Iris data,
clear
close all

fprintf('Study Fisher Iris data [R. A. Fisher "The use of multiple \n measurements in taxonomic problems" \n Annals of Eugenics 7 (1936) 179-188.]\n') 
load data_for_iris_demo.mat

%%%%Prepare inputs for DBHTs %%%%%%%%%
D = pdist(data,'cityblock');D=squareform(D); % dissimilatity matrix
S = 1./(1+D/(mean(abs(D(:))))); % similarity matrix
% ---- different choices for D, S give different outputs!
% try for instance
% D=pdist(data,'euclidean'); D=squareform(D);
% S = 2-D.^2/2;

[T8,Rpm,Adjv,Dpm,Mv,Z]=DBHTs(D,S);% DBHT clustering
fprintf('Found %d clusters \n',length(unique(T8)))

% Identifying  matching partition with iris calssification
fprintf('Analyze 3 superclusters form hierarcy\n')
Tz=cluster(Z,'MaxClust',3);
T1=Tz;T2=Tz;T3=Tz;T4=Tz;T5=Tz;T6=Tz;
T1(Tz==1)=1;T1(Tz==2)=2;T1(Tz==3)=3;
T2(Tz==1)=1;T2(Tz==2)=3;T2(Tz==3)=2;
T3(Tz==1)=2;T3(Tz==2)=1;T3(Tz==3)=3;
T4(Tz==1)=2;T4(Tz==2)=3;T4(Tz==3)=1;
T5(Tz==1)=3;T5(Tz==2)=2;T5(Tz==3)=1;
T6(Tz==1)=3;T6(Tz==2)=1;T6(Tz==3)=2;
[m,k]=min([sum([T-T1]~=0),sum([T-T2]~=0),sum([T-T3]~=0),sum([T-T4]~=0),sum([T-T5]~=0),sum([T-T6]~=0)]);
fprintf('Number of misalignments = %d over %d in total\n',m,length(T))

