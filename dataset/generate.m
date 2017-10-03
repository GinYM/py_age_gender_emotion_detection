categories = { 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise'};
for i = 1:size(categories,2)
    img = imread([categories{i},'.png']);
    save(categories{i},'img');
end