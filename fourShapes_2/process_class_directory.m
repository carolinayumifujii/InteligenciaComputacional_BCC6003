
function [] = process_class_directory(I, class_label, fid_full, fid_quad)
    % Calcular momentos invariantes de Hu para a imagem inteira
    phi_full = invmoments(I);
    write_features(fid_full, phi_full, class_label);

    % Dividir a imagem em quadrantes e calcular momentos invariantes de Hu para cada quadrante
    numQuadrants = 2;
    [rows, cols] = size(I);
    quadrant_rows = floor(rows / numQuadrants);
    quadrant_cols = floor(cols / numQuadrants);
    for r = 0:numQuadrants-1
        for c = 0:numQuadrants-1
            % Extrair o quadrante atual
            row_start = r * quadrant_rows + 1;
            row_end = (r + 1) * quadrant_rows;
            col_start = c * quadrant_cols + 1;
            col_end = (c + 1) * quadrant_cols;
            quadrant = I(row_start:row_end, col_start:col_end);
            % Calcular os momentos invariantes de Hu para o quadrante atual
            phi_quadrant = invmoments(quadrant);
            % Escrever os momentos invariantes de Hu do quadrante no arquivo
            write_features(fid_quad, phi_quadrant, class_label);
        end
    end
end

