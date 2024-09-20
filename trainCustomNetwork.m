function [net, info] = trainCustomNetwork(dlnet, XTrain, YTrain, options)
    %  dlnet = dlnetwork(Mylayers);
    % 将数据移到 GPU 上
    dlX = gpuArray(XTrain);
    dlY = gpuArray(YTrain);

    % 训练网络
    numEpochs = options.MaxEpochs;
    InitialLearnRate=options.InitialLearnRate;
    
    % 初始化 Adam 优化器的动量变量
    % Adam 优化器的参数
    decayRate = 0.001;  % 学习率衰减
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;

    % 记录损失值
    lossArray = zeros(1, numEpochs);

    % 创建训练进度监视器
    if strcmp(options.Plots, 'training-progress')
        show_flag = 1;
        monitor = trainingProgressMonitor;
        monitor.Metrics = ["TotalLoss", "MSELoss"];
        monitor.XLabel = "Epoch";
    else
        show_flag = 0;
    end
    averageGrad=[];
    averageSqGrad=[];
    % 开始训练循环
    for epoch = 1:numEpochs
        % 动态调整学习率
        learningRate = InitialLearnRate * (1 / (1 + decayRate * epoch));
        % 前向传播和损失计算，使用 dlfeval
        [gradients, loss, loss_mse] = dlfeval(@modelGradients, dlnet, dlX, dlY);
        % 遍历 gradients 表中的每个元素，并添加噪声
        for i = 1:size(gradients, 1)
            % 提取当前的梯度值
            gradValue = gradients.Value{i};
            % 生成与当前梯度维度匹配的随机噪声，并确保其类型与梯度一致  GPU or CPU
            noise = 0.5*learningRate * randn(size(gradValue), 'like', gradValue);  % 'like' 保证类型一致
            % 将噪声添加到梯度值中
            gradients.Value{i} = gradValue + noise;
        end

        % Adam Update / Adam 更新
        [dlnet, averageGrad, averageSqGrad] = ...
            adamupdate( dlnet, gradients, averageGrad, averageSqGrad,epoch,learningRate);
        % 记录损失值
        lossArray(epoch) = extractdata(loss);
        
        if show_flag
            recordMetrics(monitor, epoch, "TotalLoss", loss, "MSELoss", loss_mse);
            monitor.Progress = 100 * epoch / numEpochs;
        end
    end

    % 返回训练后的网络和损失信息
    net = dlnet;
    info = struct('Loss', lossArray);
end


function [gradients, loss, loss_mse] = modelGradients(dlnet, dlX, dlY)
    % 前向传播
     dlYpred = forward(dlnet, dlX);
    % 计算损失
    Error=dlYpred - dlY;
    loss_mse = mean((Error).^2);
    loss=mean((tanh(5*Error)).^2);%tanh(loss_mse);
    % 计算梯度
    gradients = dlgradient(loss, dlnet.Learnables);
end
