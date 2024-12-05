load('DataDemo.mat')
Mylayers=[
    featureInputLayer(9)
    lstmLayer(27,"OutputMode","sequence");
    tanhLayer
    fullyConnectedLayer(81)
    tanhLayer
    fullyConnectedLayer(27)
    tanhLayer
    fullyConnectedLayer(9)
    tanhLayer
    fullyConnectedLayer(1)
    % regressionLayer
    ];

MyOptions = trainingOptions('adam', ...
    'MaxEpochs', 600, ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 1e-3, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'L2Regularization', 0.0005, ... % 加入L2正则化
    'ExecutionEnvironment', 'gpu'); % 使用GPU加速


%% 使用自定义函数更新神经网络
Prenet=dlnetwork(Mylayers);
XTrain = dlarray(ThisDataX, 'CB'); 
YTrain = dlarray(ThisDataY, 'CB'); 
Thisnet=trainCustomNetwork(XTrain,YTrain,Mylayers,MyOptions);
ypred=forward(Thisnet,XTrain);


% ypred=predict(Thisnet,ThisDataX);
plot(ThisDataX(7,:),ThisDataY,'k');hold on;
plot(ThisDataX(7,:),ypred,'r');
rmsetotal=sqrt(mean((ThisDataY-ypred).^2));
rmse1=sqrt(mean((ThisDataY(1:446)-ypred(1:446)).^2));
disp('RMSEtotal=');
disp(rmsetotal);
disp('RMSE1=');
disp(rmse1);
