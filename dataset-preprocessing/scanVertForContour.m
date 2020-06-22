function contour = scanVertForContour(image, default, point1, point2, zone1, zone2, top)

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


for x_coord = point1:point2

    ind = x_coord - point1 + 1;
    
    % scan search zone for max change in intensity
    for y_coord = default(ind)-zone1:default(ind)+zone2

        diff(ind) = abs(double(image(inLimits(y_coord+1, imSize(1)),x_coord))-double(image(inLimits(y_coord-1, imSize(1)),x_coord)));
        
        % searching for upper pixel darker than lower for top of vertebra
        % searching for lower pixel darker than upper for bottom of vertebra
        if top && (image(inLimits(y_coord+1, imSize(1)),x_coord) < image(inLimits(y_coord-1, imSize(1)),x_coord))
        elseif ~top && (image(inLimits(y_coord+1, imSize(1)),x_coord) > image(inLimits(y_coord-1, imSize(1)),x_coord))
        elseif diff(ind) > max_diff(ind)
            max_diff(ind) = diff(ind);
            max_diff_ind(ind) = y_coord;
        end

    end
end

x_contour = point1:point2;
y_contour = max_diff_ind;

% eliminate max changes in intensity that are beyond the threshold from the
% default line

bool_arr = zeros(size(y_contour));
for point=1:line_length
    if abs(y_contour(point)- default(point)) > threshold
        bool_arr(point) = 1;
    end
end
for point=1:line_length
    if bool_arr(point) == 1
        y_contour(point) = default(point);
    end
end



% fit polynomial to max changes in intensity
p = polyfit(x_contour,y_contour,n);
x1 = linspace(min(x_contour),max(x_contour));
y_contour_smooth = polyval(p,x1);
contour = [x1; y_contour_smooth]';

end

