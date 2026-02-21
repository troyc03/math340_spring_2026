function [z, n] = compute_convergent_seq(z1)
    % Initial settings
    z = [z1];       % Start the array with the first term
    n = 1;          % Initial index
    tolerance = 1e-4; 
    difference = Inf; % Initialize difference as infinity
    
    fprintf('n \t Value \t\t Difference\n');
    fprintf('%d \t %.8f \t -\n', n, z(n));

    % Loop until the change between terms is less than 10^-4
    while difference > tolerance
        % The recurrence formula
        term_inside = 1 - (4^(1-n)) * (z(n)^2);
        
        % Precision guard
        if term_inside < 0, term_inside = 0; end
        
        % Calculate next term
        next_val = (2^n) * sqrt(0.5 * (1 - sqrt(term_inside)));
        
        % Update difference and append to sequence
        difference = abs(next_val - z(end));
        z(end+1) = next_val; 
        n = n + 1;
        
        % Print progress
        fprintf('%d \t %.8f \t %.8e\n', n, z(end), difference);
        
        % Safety break to prevent infinite loops if it doesn't converge
        if n > 1000
            warning('Maximum iterations reached.');
            break;
        end
    end

    % Run this in command window
    % [results, iterations] = compute_convergent_seq(2);

