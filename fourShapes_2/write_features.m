function [] = write_features(fid, phi, class_label)
    % Escrever os momentos invariantes de Hu no arquivo
    fprintf(fid, '%f ', phi);
    % Escrever a classe no final da linha
    fprintf(fid, '%d\n', class_label);
end