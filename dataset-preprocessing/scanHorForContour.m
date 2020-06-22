function contour = scanHorForContour(image, default, point1, point2, zone1, zone2, left)

imSize = size(image);
threshold = imSize(1)*imSize(2); % 1 -> default, big -> no correction ... imSize(1)*imSize(2)/75000 for some correction

n = 4;  % order of polynomial to fit

% length of default line
line_length = size(default);
line_length = line_length(2);

% variables to store the difference in intensity between two pixels
% also need to store the maximum difference and its index
diff = zeros(1, line_length);
max_diff = diff;
max_diff_ind = diff + default;


for y_coord = point1:point2

    ind = y_coord - point1 + 1;
    
    % scan search zone for max change in intensity
    for x_coord = default(ind)-zone1:default(ind)+zone2

        diff(ind) = abs(double(image(y_coord,inLimits(x_coord+1, imSize(2))))-double(image(y_coord,inLimits(x_coord-1, imSize(2)))));
        
        % searching for left pixel darker than right for left-side of vertebra
        % searching for right pixel darker than left for right-side of vertebra
        if left && image(y_coord,inLimits(x_coord+1, imSize(2))) < image(y_coord,inLimits(x_coord-1, imSize(2)))
        elseif ~left && image(y_coord,inLimits(x_coord+1, imSize(2))) > image(y_coord,inLimits(x_coord-1, imSize(2)))
        elseif diff(ind) > max_diff(ind)
            max_diff(ind) = diff(ind);
            max_diff_ind(ind) = x_coord;
        end

    end
end

y_contour = point1:point2;
x_contour = max_diff_ind;

% eliminate max changes in intensity that are beyond the threshold from the
% default line
bool_arr = zeros(size(x_contour));
for point=1:line_length
    if abs(x_contour(point)- default(point)) > threshold
        bool_arr(point) = 1;
    end
end
for point=1:line_length
    if bool_arr(point) == 1
        x_contour(point) = default(point);
    end
end

% fit polynomial to max changes in intensity
p = polyfit(y_contour,x_contour,n);
y1 = linspace(min(y_contour),max(y_contour));
x_contour_smooth = polyval(p,y1);
contour = [x_contour_smooth; y1]';

end

