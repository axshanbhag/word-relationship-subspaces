function outputDir = run_poster_controls(settingsFile, embeddingFile, embeddingDescription)
% Run controlled follow-up experiments without modifying the original script.
% Base MATLAB R2016b+; no additional toolboxes. Example:
% run_poster_controls('experiment_settings(3).mat','glove_100k.txt', ...
%   'Exact source unknown; locally supplied file')
% The settings MAT supplies pairGroups, maxWords and useDemo.
% Only selected pair vectors are retained, but the same vocabulary cap is read.
if nargin < 1 || isempty(settingsFile)
    [f,p]=uigetfile('*.mat','Select original experiment settings');
    assert(~isequal(f,0),'No settings file selected.');
    settingsFile=fullfile(p,f);
end
cfg=load(settingsFile);
assert(isfield(cfg,'pairGroups') && isfield(cfg,'maxWords'), ...
    'Settings must contain pairGroups and maxWords.');
assert(isfield(cfg,'useDemo') && ~cfg.useDemo, ...
    'Real-embedding settings required; synthetic settings are rejected.');
if nargin < 2 || isempty(embeddingFile)
    [f,p]=uigetfile('*.txt','Select the SAME embedding file used originally');
    assert(~isequal(f,0),'No embedding file selected.');
    embeddingFile=fullfile(p,f);
end
if nargin < 3 || isempty(embeddingDescription)
    embeddingDescription='UNVERIFIED: supply original download URL and release if known';
end
seed=20261102; permutations=1000; repeats=10; folds=5;
controlsPerFold=20; ranks=[1 2 5 10 15 20]; threshold=0.90;
% Prespecified main endpoint: correct minus shuffled held-out energy at k=5.
% Other ranks, full-data metrics and category differences are descriptive.
outputDir=tempname(fileparts(mfilename('fullpath')));
mkdir(outputDir); started=tic;
rng(seed,'twister'); initialRng=rng;
groups=cfg.pairGroups; wanted={};
for g=1:size(groups,1)
    pairs=groups{g,2}; wanted=[wanted;pairs(:)]; %#ok<AGROW>
end
wanted=unique(wanted,'stable');
[vectors,found,rowsRead,dimension]=readSelected(embeddingFile,cfg.maxWords,wanted);
assert(dimension >= max(ranks),'Embedding dimension smaller than requested ranks.');
lookup=containers.Map(wanted,num2cell(1:numel(wanted)));
coverage=cell(0,5); observed=cell(0,8); nullRows=cell(0,8);
summary=cell(0,8); cvRows=cell(0,10); splitRows=cell(0,6);
repeatRows=cell(0,7); contrastRows=cell(0,6);
for g=1:size(groups,1)
    name=groups{g,1}; pairs=groups{g,2}; valid=false(size(pairs,1),1);
    for j=1:size(pairs,1)
        a=lookup(pairs{j,1}); b=lookup(pairs{j,2});
        state='used';
        if ~found(a) || ~found(b)
            state='missing vocabulary';
        elseif norm(vectors(b,:)-vectors(a,:)) <= eps
            state='zero difference';
        else
            valid(j)=true;
        end
        coverage(end+1,:)={name,j,pairs{j,1},pairs{j,2},state}; %#ok<AGROW>
    end
    originalRows=find(valid); pairs=pairs(valid,:); n=size(pairs,1);
    assert(n>=10,'Too few valid pairs in %s.',name);
    A=zeros(n,dimension); B=A;
    for j=1:n
        A(j,:)=vectors(lookup(pairs{j,1}),:);
        B(j,:)=vectors(lookup(pairs{j,2}),:);
    end
    D=B-A; obs=metrics(D,threshold); mu=mean(D,1);
    observed(end+1,:)=[{name,n,dimension},num2cell(obs), ...
        {sum(mu.^2)/mean(sum(D.^2,2))}]; %#ok<AGROW>
    null=zeros(permutations,3);
    for b=1:permutations
        order=randperm(n); Dp=B(order,:)-A;
        % Full random permutations include fixed points; no derangement filter.
        m=metrics(Dp,threshold); null(b,:)=m(1:3);
        assert(norm(mean(Dp,1)-mu)<1e-9*max(1,norm(mu)), ...
            'Permutation should preserve the mean.');
        nullRows(end+1,:)=[{name,b,sum(order==(1:n))}, ...
            num2cell(m),{norm(mean(Dp,1)-mu)}]; %#ok<AGROW>
    end
    labels={'MeanPairwiseCosine','FirstSVDEnergyFraction','SVDK90'};
    for q=1:3
        % Two-sided Monte Carlo tail score; exploratory, no multiplicity correction.
        tailLow=(1+sum(null(:,q)<=obs(q)))/(permutations+1);
        tailHigh=(1+sum(null(:,q)>=obs(q)))/(permutations+1);
        summary(end+1,:)={name,labels{q},obs(q),mean(null(:,q)), ...
            percentile(null(:,q),2.5),percentile(null(:,q),97.5), ...
            obs(q)-mean(null(:,q)),min(1,2*min(tailLow,tailHigh))}; %#ok<AGROW>
    end
    % Connected components put all pairs sharing ANY token into the same fold.
    component=wordComponents(pairs); count=max(component);
    assert(count>=folds,'Fewer word-disjoint components than folds.');
    num=zeros(repeats,numel(ranks),3); den=zeros(repeats,1);
    for r=1:repeats
        fold=assignFolds(component,folds);
        for j=1:n
            splitRows(end+1,:)={name,r,originalRows(j),component(j),fold(j), ...
                [pairs{j,1} ' -> ' pairs{j,2}]}; %#ok<AGROW>
        end
        for f=1:folds
            te=fold==f; tr=~te; test=D(te,:); train=D(tr,:);
            assert(~any(ismember(reshape(pairs(te,:),[],1), ...
                reshape(pairs(tr,:),[],1))),'Shared-word leakage detected.');
            [~,S,V]=svd(train,'econ'); ss=diag(S);
            rankTrain=sum(ss>max(size(train))*eps(max(ss)));
            assert(rankTrain>=max(ranks),'Training rank too small; reduce rank grid.');
            testEnergy=sum(test(:).^2); den(r)=den(r)+testEnergy;
            correct=zeros(1,numel(ranks)); shuffled=correct; random=correct;
            for q=1:numel(ranks)
                Z=test*V(:,1:ranks(q)); correct(q)=sum(Z(:).^2);
            end
            for b=1:controlsPerFold
                trainB=B(tr,:); trainA=A(tr,:);
                shuffledTrain=trainB(randperm(sum(tr)),:)-trainA;
                [~,Sp,Vp]=svd(shuffledTrain,'econ'); sp=diag(Sp);
                rankNull=sum(sp>max(size(shuffledTrain))*eps(max(sp)));
                assert(rankNull>=max(ranks),'Shuffled training rank too small.');
                [Q,~]=qr(randn(dimension,max(ranks)),0);
                for q=1:numel(ranks)
                    Z=test*Vp(:,1:ranks(q));
                    shuffled(q)=shuffled(q)+sum(Z(:).^2)/controlsPerFold;
                    Z=test*Q(:,1:ranks(q));
                    random(q)=random(q)+sum(Z(:).^2)/controlsPerFold;
                end
            end
            for q=1:numel(ranks)
                num(r,q,1)=num(r,q,1)+correct(q);
                num(r,q,2)=num(r,q,2)+shuffled(q);
                num(r,q,3)=num(r,q,3)+random(q);
                cvRows(end+1,:)={name,r,f,ranks(q),sum(tr),sum(te), ...
                    testEnergy,correct(q)/testEnergy, ...
                    shuffled(q)/testEnergy,random(q)/testEnergy}; %#ok<AGROW>
            end
        end
        for q=1:numel(ranks)
            vals=reshape(num(r,q,:),1,3)/den(r);
            assert(all(vals>=-1e-10 & vals<=1+1e-10),'Invalid retained fraction.');
            repeatRows(end+1,:)=[{name,r,ranks(q)},num2cell(vals), ...
                {vals(1)-vals(2)}]; %#ok<AGROW>
        end
    end
    for q=1:numel(ranks)
        delta=reshape(num(:,q,1)-num(:,q,2),[],1)./den;
        contrastRows(end+1,:)={name,ranks(q),mean(delta), ...
            min(delta),max(delta),std(delta)}; %#ok<AGROW>
    end
    fprintf('Finished %s (%d pairs); %.1f seconds elapsed.\n',name,n,toc(started));
end
writeRows(coverage,{'Relationship','OriginalPairRow','Source','Target','Status'},'coverage.csv');
writeRows(observed,{'Relationship','ValidPairs','EmbeddingDimension','MeanPairwiseCosine', ...
    'FirstSVDEnergyFraction','SVDK90','PCAK90','MeanEnergyFraction'},'observed.csv');
writeRows(nullRows,{'Relationship','Permutation','FixedIndexMatches','MeanPairwiseCosine', ...
    'FirstSVDEnergyFraction','SVDK90','PCAK90','MeanDifferenceNorm'},'permutation_draws.csv');
writeRows(summary,{'Relationship','Metric','Observed','NullMean','NullP025','NullP975', ...
    'ObservedMinusNullMean','ExploratoryTwoSidedTailScore'},'permutation_summary.csv');
writeRows(splitRows,{'Relationship','Repeat','OriginalPairRow','Component','Fold','Pair'},'splits.csv');
writeRows(cvRows,{'Relationship','Repeat','Fold','K','TrainPairs','TestPairs','TestEnergy', ...
    'CorrectRetained','ShuffledRetained','RandomRetained'},'heldout_folds.csv');
writeRows(repeatRows,{'Relationship','Repeat','K','CorrectRetained','ShuffledRetained', ...
    'RandomRetained','CorrectMinusShuffled'},'heldout_repeats.csv');
writeRows(contrastRows,{'Relationship','K','MeanCorrectMinusShuffled','MinAcrossSplits', ...
    'MaxAcrossSplits','SDAcrossSplits'},'heldout_contrasts.csv');
fileInfo=dir(embeddingFile); matlabVersion=version; elapsedSeconds=toc(started);
save(fullfile(outputDir,'run_settings.mat'),'cfg','settingsFile','embeddingFile', ...
    'embeddingDescription','fileInfo','rowsRead','dimension','seed','initialRng', ...
    'permutations','repeats','folds','controlsPerFold','ranks','threshold', ...
    'matlabVersion','elapsedSeconds');
copyfile([mfilename('fullpath') '.m'],fullfile(outputDir,'run_poster_controls.m'));
fprintf('Upload this entire results folder: %s\n',outputDir);
    function writeRows(rows,names,filename)
        writetable(cell2table(rows,'VariableNames',names),fullfile(outputDir,filename));
    end
end

function m=metrics(D,threshold)
lengths=sqrt(sum(D.^2,2));
assert(all(lengths>eps),'A permutation produced a zero arrow; inspect word lists.');
U=bsxfun(@rdivide,D,lengths); n=size(D,1);
% Sum of off-diagonal cosines, without allocating all pairwise comparisons.
alignment=(sum(sum(U,1).^2)-n)/(n*(n-1));
s=svd(D,'econ').^2; c=cumsum(s)/sum(s);
X=bsxfun(@minus,D,mean(D,1)); sc=svd(X,'econ').^2;
if sum(sc)>eps*max(1,sum(D(:).^2))
    pc=find(cumsum(sc)/sum(sc)>=threshold,1);
else
    pc=0;
end
m=[alignment,s(1)/sum(s),find(c>=threshold,1),pc];
end

function component=wordComponents(pairs)
n=size(pairs,1); component=(1:n)';
for a=1:n
    for b=a+1:n
        if any(ismember(pairs(a,:),pairs(b,:)))
            old=component(b); component(component==old)=component(a);
        end
    end
end
[~,~,component]=unique(component);
end

function fold=assignFolds(component,K)
sizes=accumarray(component,1); order=randperm(numel(sizes));
[~,ix]=sort(sizes(order),'descend'); order=order(ix);
loads=zeros(K,1); byComponent=zeros(numel(sizes),1);
for j=order
    choices=find(loads==min(loads)); f=choices(randi(numel(choices)));
    byComponent(j)=f; loads(f)=loads(f)+sizes(j);
end
fold=byComponent(component);
end

function q=percentile(x,p)
x=sort(x(:)); z=1+(numel(x)-1)*p/100;
q=x(floor(z))+(z-floor(z))*(x(ceil(z))-x(floor(z)));
end

function [E,found,n,d]=readSelected(path,maxWords,wanted)
assert(exist(path,'file')==2,'Embedding file not found.');
fid=fopen(path,'r'); assert(fid~=-1,'Cannot open embedding file.');
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
index=containers.Map(wanted,num2cell(1:numel(wanted)));
found=false(numel(wanted),1); E=[]; n=0; d=[]; first=true;
while n<maxWords
    line=fgetl(fid); if ~ischar(line),break;end
    if isempty(strtrim(line)),continue;end
    if first
        first=false;
        if ~isempty(regexp(line,'^\d+\s+\d+$','once')),continue;end
    end
    split=regexp(line,'[ \t]','once');
    assert(~isempty(split),'Malformed embedding row.');
    token=line(1:split-1); values=sscanf(line(split+1:end),'%f')';
    if isempty(d)
        d=numel(values); assert(d>=2,'Invalid dimension.');
        E=zeros(numel(wanted),d);
    end
    assert(numel(values)==d && all(isfinite(values)), ...
        'Invalid embedding row %d.',n+1);
    n=n+1;
    if isKey(index,token)
        j=index(token);
        if ~found(j) && norm(values)>0
            E(j,:)=values; found(j)=true;
        end
    end
end
assert(n>0,'No embedding rows found.');
end