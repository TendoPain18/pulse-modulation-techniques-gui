for i = 1:20
    if i == 10
        continue; % Skip iteration 10
    end
    
    if i == 14
        i = 15; % Directly jump to iteration 15
    end
    
    disp(i);
end
