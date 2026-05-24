CREATE NODE TABLE nodes(id INT64, club STRING, PRIMARY KEY(id)) WITH (storage = '', format = 'icebug-disk');
CREATE REL TABLE edges(FROM nodes TO nodes) WITH (storage = '', format = 'icebug-disk');
