# js = {"func": "    public static void main(String[] args) {\n        LogFrame.getInstance();\n        for (int i = 0; i < args.length; i++) {\n            String arg = args[i];\n            if (arg.trim().startsWith(DEBUG_PARAMETER_NAME + \"=\")) {\n                properties.put(DEBUG_PARAMETER_NAME, arg.trim().substring(DEBUG_PARAMETER_NAME.length() + 1).trim());\n                if (properties.getProperty(DEBUG_PARAMETER_NAME).toLowerCase().equals(DEBUG_TRUE)) {\n                    DEBUG = true;\n                }\n            } else if (arg.trim().startsWith(MODE_PARAMETER_NAME + \"=\")) {\n                properties.put(MODE_PARAMETER_NAME, arg.trim().substring(MODE_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(AUTOCONNECT_PARAMETER_NAME + \"=\")) {\n                properties.put(AUTOCONNECT_PARAMETER_NAME, arg.trim().substring(AUTOCONNECT_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(SITE_CONFIG_URL_PARAMETER_NAME + \"=\")) {\n                properties.put(SITE_CONFIG_URL_PARAMETER_NAME, arg.trim().substring(SITE_CONFIG_URL_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(LOAD_PLUGINS_PARAMETER_NAME + \"=\")) {\n                properties.put(LOAD_PLUGINS_PARAMETER_NAME, arg.trim().substring(LOAD_PLUGINS_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(ONTOLOGY_URL_PARAMETER_NAME + \"=\")) {\n                properties.put(ONTOLOGY_URL_PARAMETER_NAME, arg.trim().substring(ONTOLOGY_URL_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(REPOSITORY_PARAMETER_NAME + \"=\")) {\n                properties.put(REPOSITORY_PARAMETER_NAME, arg.trim().substring(REPOSITORY_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(ONTOLOGY_TYPE_PARAMETER_NAME + \"=\")) {\n                properties.put(ONTOLOGY_TYPE_PARAMETER_NAME, arg.trim().substring(ONTOLOGY_TYPE_PARAMETER_NAME.length() + 1).trim());\n                if (!(properties.getProperty(ONTOLOGY_TYPE_PARAMETER_NAME).equals(ONTOLOGY_TYPE_RDFXML) || properties.getProperty(ONTOLOGY_TYPE_PARAMETER_NAME).equals(ONTOLOGY_TYPE_TURTLE) || properties.getProperty(ONTOLOGY_TYPE_PARAMETER_NAME).equals(ONTOLOGY_TYPE_NTRIPPLES))) System.out.println(\"WARNING! Unknown ontology type: '\" + properties.getProperty(ONTOLOGY_TYPE_PARAMETER_NAME) + \"' (Known types are: '\" + ONTOLOGY_TYPE_RDFXML + \"', '\" + ONTOLOGY_TYPE_TURTLE + \"', '\" + ONTOLOGY_TYPE_NTRIPPLES + \"')\");\n            } else if (arg.trim().startsWith(OWLIMSERVICE_URL_PARAMETER_NAME + \"=\")) {\n                properties.put(OWLIMSERVICE_URL_PARAMETER_NAME, arg.trim().substring(OWLIMSERVICE_URL_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(DOCSERVICE_URL_PARAMETER_NAME + \"=\")) {\n                properties.put(DOCSERVICE_URL_PARAMETER_NAME, arg.trim().substring(DOCSERVICE_URL_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(DOC_ID_PARAMETER_NAME + \"=\")) {\n                properties.put(DOC_ID_PARAMETER_NAME, arg.trim().substring(DOC_ID_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(ANNSET_NAME_PARAMETER_NAME + \"=\")) {\n                properties.put(ANNSET_NAME_PARAMETER_NAME, arg.trim().substring(ANNSET_NAME_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(EXECUTIVE_SERVICE_URL_PARAMETER_NAME + \"=\")) {\n                properties.put(EXECUTIVE_SERVICE_URL_PARAMETER_NAME, arg.trim().substring(EXECUTIVE_SERVICE_URL_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(USER_ID_PARAMETER_NAME + \"=\")) {\n                properties.put(USER_ID_PARAMETER_NAME, arg.trim().substring(USER_ID_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(USER_PASSWORD_PARAMETER_NAME + \"=\")) {\n                properties.put(USER_PASSWORD_PARAMETER_NAME, arg.trim().substring(USER_PASSWORD_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(EXECUTIVE_PROXY_FACTORY_PARAMETER_NAME + \"=\")) {\n                properties.put(EXECUTIVE_PROXY_FACTORY_PARAMETER_NAME, arg.trim().substring(EXECUTIVE_PROXY_FACTORY_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(DOCSERVICE_PROXY_FACTORY_PARAMETER_NAME + \"=\")) {\n                properties.put(DOCSERVICE_PROXY_FACTORY_PARAMETER_NAME, arg.trim().substring(DOCSERVICE_PROXY_FACTORY_PARAMETER_NAME.length() + 1).trim());\n                RichUIUtils.setDocServiceProxyFactoryClassname(properties.getProperty(DOCSERVICE_PROXY_FACTORY_PARAMETER_NAME));\n            } else if (arg.trim().startsWith(LOAD_ANN_SCHEMAS_NAME + \"=\")) {\n                properties.put(LOAD_ANN_SCHEMAS_NAME, arg.trim().substring(LOAD_ANN_SCHEMAS_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(SELECT_AS_PARAMETER_NAME + \"=\")) {\n                properties.put(SELECT_AS_PARAMETER_NAME, arg.trim().substring(SELECT_AS_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(SELECT_ANN_TYPES_PARAMETER_NAME + \"=\")) {\n                properties.put(SELECT_ANN_TYPES_PARAMETER_NAME, arg.trim().substring(SELECT_ANN_TYPES_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(ENABLE_ONTOLOGY_EDITOR_PARAMETER_NAME + \"=\")) {\n                properties.put(ENABLE_ONTOLOGY_EDITOR_PARAMETER_NAME, arg.trim().substring(ENABLE_ONTOLOGY_EDITOR_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(CLASSES_TO_HIDE_PARAMETER_NAME + \"=\")) {\n                properties.put(CLASSES_TO_HIDE_PARAMETER_NAME, arg.trim().substring(CLASSES_TO_HIDE_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(CLASSES_TO_SHOW_PARAMETER_NAME + \"=\")) {\n                properties.put(CLASSES_TO_SHOW_PARAMETER_NAME, arg.trim().substring(CLASSES_TO_SHOW_PARAMETER_NAME.length() + 1).trim());\n            } else if (arg.trim().startsWith(ENABLE_APPLICATION_LOG_PARAMETER_NAME + \"=\")) {\n                properties.put(ENABLE_APPLICATION_LOG_PARAMETER_NAME, arg.trim().substring(ENABLE_APPLICATION_LOG_PARAMETER_NAME.length() + 1).trim());\n            } else {\n                System.out.println(\"WARNING! Unknown or undefined parameter: '\" + arg.trim() + \"'\");\n            }\n        }\n        System.out.println(startupParamsToString());\n        if (properties.getProperty(MODE_PARAMETER_NAME) == null || (!(properties.getProperty(MODE_PARAMETER_NAME).toLowerCase().equals(POOL_MODE)) && !(properties.getProperty(MODE_PARAMETER_NAME).toLowerCase().equals(DIRECT_MODE)))) {\n            String err = \"Mandatory parameter '\" + MODE_PARAMETER_NAME + \"' must be defined and must have a value either '\" + POOL_MODE + \"' or '\" + DIRECT_MODE + \"'.\\n\\nApplication will exit.\";\n            System.out.println(err);\n            JOptionPane.showMessageDialog(new JFrame(), err, \"Error!\", JOptionPane.ERROR_MESSAGE);\n            System.exit(-1);\n        }\n        if (properties.getProperty(SITE_CONFIG_URL_PARAMETER_NAME) == null || properties.getProperty(SITE_CONFIG_URL_PARAMETER_NAME).length() == 0) {\n            String err = \"Mandatory parameter '\" + SITE_CONFIG_URL_PARAMETER_NAME + \"' is missing.\\n\\nApplication will exit.\";\n            System.out.println(err);\n            JOptionPane.showMessageDialog(new JFrame(), err, \"Error!\", JOptionPane.ERROR_MESSAGE);\n            System.exit(-1);\n        }\n        try {\n            String context = System.getProperty(CONTEXT);\n            if (context == null || \"\".equals(context)) {\n                context = DEFAULT_CONTEXT;\n            }\n            String s = System.getProperty(GateConstants.GATE_HOME_PROPERTY_NAME);\n            if (s == null || s.length() == 0) {\n                File f = File.createTempFile(\"foo\", \"\");\n                String gateHome = f.getParent().toString() + context;\n                f.delete();\n                System.setProperty(GateConstants.GATE_HOME_PROPERTY_NAME, gateHome);\n                f = new File(System.getProperty(GateConstants.GATE_HOME_PROPERTY_NAME));\n                if (!f.exists()) {\n                    f.mkdirs();\n                }\n            }\n            s = System.getProperty(GateConstants.PLUGINS_HOME_PROPERTY_NAME);\n            if (s == null || s.length() == 0) {\n                System.setProperty(GateConstants.PLUGINS_HOME_PROPERTY_NAME, System.getProperty(GateConstants.GATE_HOME_PROPERTY_NAME) + \"/plugins\");\n                File f = new File(System.getProperty(GateConstants.PLUGINS_HOME_PROPERTY_NAME));\n                if (!f.exists()) {\n                    f.mkdirs();\n                }\n            }\n            s = System.getProperty(GateConstants.GATE_SITE_CONFIG_PROPERTY_NAME);\n            if (s == null || s.length() == 0) {\n                System.setProperty(GateConstants.GATE_SITE_CONFIG_PROPERTY_NAME, System.getProperty(GateConstants.GATE_HOME_PROPERTY_NAME) + \"/gate.xml\");\n            }\n            if (properties.getProperty(SITE_CONFIG_URL_PARAMETER_NAME) != null && properties.getProperty(SITE_CONFIG_URL_PARAMETER_NAME).length() > 0) {\n                File f = new File(System.getProperty(GateConstants.GATE_SITE_CONFIG_PROPERTY_NAME));\n                if (f.exists()) {\n                    f.delete();\n                }\n                f.getParentFile().mkdirs();\n                f.createNewFile();\n                URL url = new URL(properties.getProperty(SITE_CONFIG_URL_PARAMETER_NAME));\n                InputStream is = url.openStream();\n                FileOutputStream fos = new FileOutputStream(f);\n                int i = is.read();\n                while (i != -1) {\n                    fos.write(i);\n                    i = is.read();\n                }\n                fos.close();\n                is.close();\n            }\n            try {\n                Gate.init();\n                gate.Main.applyUserPreferences();\n            } catch (Exception e) {\n                e.printStackTrace();\n            }\n            s = BASE_PLUGIN_NAME + \",\" + properties.getProperty(LOAD_PLUGINS_PARAMETER_NAME);\n            System.out.println(\"Loading plugins: \" + s);\n            loadPlugins(s, true);\n            loadAnnotationSchemas(properties.getProperty(LOAD_ANN_SCHEMAS_NAME), true);\n        } catch (Throwable e) {\n            e.printStackTrace();\n        }\n        MainFrame.getInstance().setVisible(true);\n        MainFrame.getInstance().pack();\n        if (properties.getProperty(MODE_PARAMETER_NAME).toLowerCase().equals(DIRECT_MODE)) {\n            if (properties.getProperty(AUTOCONNECT_PARAMETER_NAME, \"\").toLowerCase().equals(AUTOCONNECT_TRUE)) {\n                if (properties.getProperty(DOC_ID_PARAMETER_NAME) == null || properties.getProperty(DOC_ID_PARAMETER_NAME).length() == 0) {\n                    String err = \"Can't autoconnect. A parameter '\" + DOC_ID_PARAMETER_NAME + \"' is missing.\";\n                    System.out.println(err);\n                    JOptionPane.showMessageDialog(MainFrame.getInstance(), err, \"Error!\", JOptionPane.ERROR_MESSAGE);\n                    ActionShowDocserviceConnectDialog.getInstance().actionPerformed(null);\n                } else {\n                    ActionConnectToDocservice.getInstance().actionPerformed(null);\n                }\n            } else {\n                ActionShowDocserviceConnectDialog.getInstance().actionPerformed(null);\n            }\n        } else {\n            if (properties.getProperty(AUTOCONNECT_PARAMETER_NAME, \"\").toLowerCase().equals(AUTOCONNECT_TRUE)) {\n                if (properties.getProperty(USER_ID_PARAMETER_NAME) == null || properties.getProperty(USER_ID_PARAMETER_NAME).length() == 0) {\n                    String err = \"Can't autoconnect. A parameter '\" + USER_ID_PARAMETER_NAME + \"' is missing.\";\n                    System.out.println(err);\n                    JOptionPane.showMessageDialog(MainFrame.getInstance(), err, \"Error!\", JOptionPane.ERROR_MESSAGE);\n                    ActionShowExecutiveConnectDialog.getInstance().actionPerformed(null);\n                } else {\n                    ActionConnectToExecutive.getInstance().actionPerformed(null);\n                }\n            } else {\n                ActionShowExecutiveConnectDialog.getInstance().actionPerformed(null);\n            }\n        }\n    }\n", "idx": "10000832"}
# js2 = {"index": "3142", "code": "import java.util.*;\nimport java.io.InputStreamReader;\nimport java.io.BufferedReader;\n\n\npublic class Main {\n    public static void main(String[] args) throws Exception {\n        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));\n        String s = br.readLine();\n        \n        int count1 = 0;\n        for (int i = 0; i < s.length(); i++) {\n            if (i % 2 == 0) {\n                if (s.charAt(i) != '0') count1++;\n            }\n            else {\n                if (s.charAt(i) != '1') count1++;\n            }\n        }\n        \n        int count2 = 0;\n        for (int i = 0; i < s.length(); i++) {\n            if (i % 2 == 0) {\n                if (s.charAt(i) != '1') count2++;\n            }\n            else {\n                if (s.charAt(i) != '0') count2++;\n            }\n        }\n        \n        //System.out.println(count2);\n        System.out.println(Math.min(count1, count2));\n    }\n}\n\n\n\n", "label": 10}
#
# from tree_sitter import Language, Parser
# from utils import remove_comments_and_docstrings, get_token
#
#
# def parse_program1(func):
#     func = func.replace("\\n", "\n")
#     try:
#         func = remove_comments_and_docstrings(func, 'java')
#     except:
#         pass
#     parser = Parser()
#     parser.set_language(Language("../../parser/parser_folder/my-languages.so", 'java'))
#     tree = parser.parse(bytes(func, 'utf8'))
#     root = tree.root_node
#     return root
#
#
# import javalang
# def parse_program(func):
#     tree = javalang.parse.parse(func)
#     # tokens = javalang.tokenizer.tokenize(func)
#     # parser = javalang.parser.Parser(tokens)
#     # tree = parser.parse_member_declaration()
#     return tree
#
# #
# # # from prepare_data_java import get_sequence as func
# # from utils import get_sequence as func
# # def trans_to_sequences(ast):
# #     sequence = []
# #     func(ast, sequence)
# #     return sequence
# #
# # # root = parse_program(js["func"])
# # root = parse_program1(js2["code"])
# # print(type(root))
# # print(root)
# # seq = trans_to_sequences(root)
# # print(seq)
# #
# # root = parse_program(js2["code"])
# # print(type(root))
# #
# # from prepare_data_java import get_sequence as func
# # def trans_to_sequences(ast):
# #     sequence = []
# #     func(ast, sequence)
# #     return sequence
# # seq = trans_to_sequences(root)
# # print(seq)
#
# from utils import get_blocks as func
#
# print("...............................................................................")
# def tree_to_index(node):
#     token = node.token
#     result = [token]
#     children = node.children
#     for child in children:
#         result.append(tree_to_index(child))
#         # if "body" not in str(child.type) and "block" not in str(child.type):
#         #     result.append(tree_to_index(child))
#
#     return result
#
#
# def trans2seq(r):
#     blocks = []
#     func(r, blocks)
#     print(blocks)
#     print([b.token for b in blocks])
#     tree = []
#     for b in blocks:
#         btree = tree_to_index(b)
#         tree.append(btree)
#     return tree
#
# root = parse_program1(js2["code"])
#
# print(root.sexp())
# print("...............................................................................")
#
# for i in trans2seq(root):
#     print(i)
#
# print("...............................................................................")
#
# # from prepare_data_java import get_blocks_v1 as func
# #
# #
# # def tree_to_index(node):
# #     token = node.token
# #     result = [token]
# #     children = node.children
# #     for child in children:
# #         result.append(tree_to_index(child))
# #     return result
# #
# #
# # def trans2seq(r):
# #     blocks = []
# #     func(r, blocks)
# #     tree = []
# #     for b in blocks:
# #         print("b: ", b)
# #         btree = tree_to_index(b)
# #         tree.append(btree)
# #     return tree
# #
# # root = parse_program(js2["code"])
# #
# # for i in trans2seq(root):
# #     print(i)


