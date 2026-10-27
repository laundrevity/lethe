// Exercise the same schema, Qwen grammar, and response parser as llama-server,
// without a model or network listener. All templates and fixtures arrive on stdin.
#include "chat.h"
#include "json.h"
#include "llama-grammar.h"
#include "parsers/parsers.h"
#include "src/unicode.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

static bool accepts(const std::string & grammar_text, const std::string & text) {
    std::unique_ptr<llama_grammar> grammar(
        llama_grammar_init_impl(nullptr, grammar_text.c_str(), "root", false, nullptr, 0, nullptr, 0));
    if (!grammar) {
        throw std::runtime_error("could not compile generated grammar");
    }
    for (auto point : unicode_cpts_from_utf8(text)) {
        llama_grammar_accept(grammar.get(), point);
        if (llama_grammar_get_stacks(grammar.get()).empty()) {
            return false;
        }
    }
    const auto & stacks = llama_grammar_get_stacks(grammar.get());
    return std::any_of(stacks.begin(), stacks.end(), [](const auto & stack) { return stack.empty(); });
}

int main() {
    try {
        const auto input = common_json::parse(std::string(std::istreambuf_iterator<char>(std::cin), {}));
        const auto & function = input.at("tool").at("function");
        common_json output = common_json::object();
        output["parameters"] = common_json::array();
        foreach_parameter(function, [&](size_t, const common_chat_schema_property & parameter,
                                        const common_chat_schema_document_ptr &) {
            output["parameters"].push_back(parameter.name);
        });

        auto templates = common_chat_templates_init(nullptr, input.at("template").get<std::string>());
        common_chat_templates_inputs request;
        request.messages = {{"user", "Run the specified SSH tool."}};
        request.tools = {{function.at("name").get<std::string>(),
                          function.at("description").get<std::string>(), function.at("parameters").dump()}};
        request.enable_thinking = false;
        request.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        request.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        const auto generated = common_chat_templates_apply(templates.get(), request);
        common_chat_parser_params parser_params(generated);
        output["cases"] = common_json::array();
        for (const auto & fixture : input.at("cases")) {
            const auto text = fixture.get<std::string>();
            common_json result = common_json::object();
            result["grammar_accepts"] = accepts(generated.grammar, generated.generation_prompt + text);
            try {
                const auto parsed = common_chat_peg_parse(generated.parser, common_chat_input(text), false, parser_params);
                result["calls"] = common_json::array();
                for (const auto & call : parsed.tool_calls) {
                    common_json value = common_json::object();
                    value["name"] = call.name;
                    value["arguments"] = common_json::parse(call.arguments);
                    result["calls"].push_back(value);
                }
            } catch (const std::exception & error) {
                result["parse_error"] = error.what();
            }
            output["cases"].push_back(result);
        }
        std::cout << output.dump() << '\n';
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
