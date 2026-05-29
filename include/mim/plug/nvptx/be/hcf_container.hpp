// SPDX-License-Identifier: BSD-2-Clause AND MIT
//
// Vendored from AdaptiveCpp (BSD 2-Clause):
//   AdaptiveCpp/include/hipSYCL/common/hcf_container.hpp
// Copyright The AdaptiveCpp Contributors.
//
// MimIR-local modifications: debug-logging macros stubbed to std::cerr to drop
// the dependency on AdaptiveCpp's runtime/application.hpp. Format wire-compat
// with libacpp-rt's hcf_container parser is preserved verbatim — node start/end
// markers, binary appendix sentinel, key=value lines, all unchanged.
//
// AdaptiveCpp BSD 2-Clause license terms apply to this file in addition to
// MimIR's MIT license. See AdaptiveCpp/LICENSE for the full BSD text.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mim::ll::pcuda::hcf {

class hcf_container {
public:
    struct node {
        std::vector<std::pair<std::string, std::string>> key_value_pairs;
        std::vector<node> subnodes;
        std::string node_id;

        const node* get_subnode(const std::string& name) const {
            for (size_t i = 0; i < subnodes.size(); ++i)
                if (subnodes[i].node_id == name) return &subnodes[i];
            return nullptr;
        }
        node* get_subnode(const std::string& name) {
            for (size_t i = 0; i < subnodes.size(); ++i)
                if (subnodes[i].node_id == name) return &subnodes[i];
            return nullptr;
        }
        const std::string* get_value(const std::string& key) const {
            for (size_t i = 0; i < key_value_pairs.size(); ++i)
                if (key_value_pairs[i].first == key) return &key_value_pairs[i].second;
            return nullptr;
        }
        bool has_key(const std::string& key) const { return get_value(key) != nullptr; }
        bool has_subnode(const std::string& name) const { return get_subnode(name) != nullptr; }
        std::vector<std::string> get_subnodes() const {
            std::vector<std::string> r;
            for (const auto& s : subnodes) r.push_back(s.node_id);
            return r;
        }
        bool has_binary_data_attached() const { return has_subnode("__binary"); }
        bool is_binary_content() const { return node_id == "__binary"; }

        node* add_subnode(const std::string& unique_name) {
            for (size_t i = 0; i < subnodes.size(); ++i) {
                if (subnodes[i].node_id == unique_name) {
                    std::cerr << "hcf: Subnode already exists with name " << unique_name << "\n";
                    return nullptr;
                }
            }
            node n;
            n.node_id = unique_name;
            subnodes.push_back(n);
            return &subnodes.back();
        }
        void set(const std::string& key, const std::string& value) {
            key_value_pairs.emplace_back(key, value);
        }
        void set_as_list(const std::string& key, const std::vector<std::string>& list_entries) {
            auto* N = add_subnode(key);
            if (N)
                for (const auto& e : list_entries) N->add_subnode(e);
        }
        std::vector<std::string> get_as_list(const std::string& key) const {
            if (!has_subnode(key)) return {};
            return get_subnode(key)->get_subnodes();
        }
    };

    hcf_container() { _root_node.node_id = "root"; }

    explicit hcf_container(const std::string& container) {
        std::string appendix_id{_binary_appendix_id};
        std::size_t appendix_begin = container.find(appendix_id);
        if (appendix_begin != std::string::npos)
            _binary_appendix = container.substr(appendix_begin + appendix_id.length());
        std::string parseable_data = container;
        if (appendix_begin != std::string::npos) parseable_data.erase(appendix_begin);
        parse(parseable_data);
    }

    const node* root_node() const { return &_root_node; }
    node* root_node() { return &_root_node; }

    bool get_binary_attachment(const node* n, std::string& out) const {
        if (!n) return false;
        const node* descriptor_node = nullptr;
        if (n->is_binary_content()) descriptor_node = n;
        else if (n->has_binary_data_attached()) descriptor_node = n->get_subnode("__binary");
        else {
            std::cerr << "hcf: Node " << n->node_id
                      << " is not a binary content node, nor does it carry a binary attachment\n";
            return false;
        }
        assert(descriptor_node);
        const std::string* start_entry = descriptor_node->get_value("start");
        const std::string* size_entry  = descriptor_node->get_value("size");
        if (!start_entry || !size_entry) {
            std::cerr << "hcf: Node missing binary start/size\n";
            return false;
        }
        std::size_t start = std::stoull(*start_entry);
        std::size_t size  = std::stoull(*size_entry);
        if (start + size > _binary_appendix.size()) {
            std::cerr << "hcf: Binary content address is out-of-bounds\n";
            return false;
        }
        out = _binary_appendix.substr(start, size);
        return true;
    }

    bool attach_binary_content(node* n, const std::string& binary_content) {
        node* binary_node = n->add_subnode(_binary_marker);
        if (!binary_node) return false;
        std::size_t start  = _binary_appendix.size();
        std::size_t length = binary_content.size();
        _binary_appendix += binary_content;
        binary_node->set("start", std::to_string(start));
        binary_node->set("size", std::to_string(length));
        return true;
    }

    std::string serialize() const {
        std::stringstream sstr;
        serialize_node(_root_node, sstr);
        sstr << _binary_appendix_id;
        return sstr.str() + _binary_appendix;
    }

private:
    void serialize_node(const node& n, std::ostream& out) const {
        for (const auto& p : n.key_value_pairs) out << p.first << "=" << p.second << "\n";
        for (const auto& s : n.subnodes) {
            out << _node_start_id << s.node_id << "\n";
            serialize_node(s, out);
            out << _node_end_id << s.node_id << "\n";
        }
    }

    static void trim_left(std::string& s) {
        auto it = std::find_if(s.begin(), s.end(),
                               [](char ch) { return !std::isspace<char>(ch, std::locale::classic()); });
        s.erase(s.begin(), it);
    }
    static void trim_right(std::string& s) {
        auto it = std::find_if(s.rbegin(), s.rend(),
                               [](char ch) { return !std::isspace<char>(ch, std::locale::classic()); });
        s.erase(it.base(), s.end());
    }

    bool parse_node_start(const std::string& line, std::string& node_id) const {
        std::string proc = line;
        std::string ns{_node_start_id};
        trim_left(proc);
        if (proc.find(ns) != 0) {
            std::cerr << "hcf: Invalid node start: " << proc << "\n";
            return false;
        }
        proc.erase(0, ns.length());
        trim_right(proc);
        node_id = proc;
        return true;
    }

    bool parse_node_interior(const std::vector<std::string>& lines,
                             std::size_t node_start_line, std::size_t node_end_line,
                             node& current) const {
        if (node_start_line == node_end_line) return true;
        for (std::size_t i = node_start_line; i < node_end_line; ++i) {
            assert(i < lines.size());
            const std::string& cur = lines[i];
            if (cur.find(_node_start_id) == 0) {
                node nn;
                if (!parse_node_start(cur, nn.node_id)) return false;
                std::size_t num_lines = std::string::npos;
                std::string end_marker = std::string{_node_end_id} + nn.node_id;
                for (std::size_t j = i + 1; j < node_end_line; ++j) {
                    if (lines[j] == end_marker) {
                        num_lines = j - i;
                        break;
                    }
                }
                if (num_lines == std::string::npos) {
                    std::cerr << "hcf: Syntax error: missing node end marker: " << end_marker << "\n";
                    return false;
                }
                if (!parse_node_interior(lines, i + 1, i + num_lines, nn)) return false;
                current.subnodes.push_back(nn);
                i += num_lines;
            } else if (cur.find('=') != std::string::npos) {
                std::size_t pos = cur.find('=');
                current.key_value_pairs.emplace_back(cur.substr(0, pos), cur.substr(pos + 1));
            } else if (cur.find(_node_end_id) == 0) {
                std::cerr << "hcf: Syntax error: unexpected node end: " << cur << "\n";
                return false;
            } else {
                std::cerr << "hcf: Syntax error: invalid line: " << cur << "\n";
                return false;
            }
        }
        return true;
    }

    bool parse(const std::string& data) {
        std::stringstream ss(data);
        std::string line;
        std::vector<std::string> lines;
        while (std::getline(ss, line)) {
            trim_left(line);
            trim_right(line);
            if (!line.empty()) lines.push_back(line);
        }
        _root_node.node_id = "root";
        return parse_node_interior(lines, 0, lines.size(), _root_node);
    }

    static constexpr char _binary_appendix_id[] = "__acpp_hcf_binary_appendix";
    static constexpr char _node_start_id[]      = "{.";
    static constexpr char _node_end_id[]        = "}.";
    static constexpr char _binary_marker[]      = "__binary";

    node _root_node;
    std::string _binary_appendix;
};

} // namespace mim::ll::pcuda::hcf
