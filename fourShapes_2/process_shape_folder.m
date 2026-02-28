function [] = process_shape_folder(shape_folder, class_label)
    % Arquivo onde os momentos invariantes de Hu serão salvos
    full_image_output_file = 'momentos_invariantes_completos.txt';
    quadrant_output_file = 'momentos_invariantes_quadrantes.txt';

    % Lista os arquivos PNG na pasta da forma
    fileList = dir(fullfile(shape_folder, '*.png'));

    % Verifica se a pasta está vazia
    if isempty(fileList)
        warning('No PNG files found in folder: %s', shape_folder);
        return;
    end

    % Abre os arquivos para escrita
    fid_full = fopen(full_image_output_file, 'a'); % 'a' para adicionar ao arquivo existente
    fid_quad = fopen(quadrant_output_file, 'a'); % 'a' para adicionar ao arquivo existente

    % Loop através dos arquivos PNG
    for i = 1:length(fileList)
        % Caminho completo para o arquivo PNG atual
        img_path = fullfile(shape_folder, fileList(i).name);
        fprintf('Processing file: %s\n', img_path);
        try
            % Tenta ler a imagem PNG
            I = imread(img_path);
        catch ME
            warning('Could not read file: %s. Skipping...', img_path);
            continue;
        end
        % Redimensiona a imagem para 28x28 pixels
        I = imresize(I, [28, 28]);
        % Processa o diretório de classe atual para calcular e escrever os momentos invariantes de Hu
        process_class_directory(I, class_label, fid_full, fid_quad);
    end

    % Fecha os arquivos após o processamento da pasta de formas atual
    fclose(fid_full);
    fclose(fid_quad);
end
